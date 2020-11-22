#include <iostream>
#include <algorithm>
#include <limits>
#include <random>

#include "opencl_neural_network.h"

cl_int opencl_feed_forward(OpenCL_Data& opencl, cl::Kernel& kernel, cl::Buffer& previous_layer, cl::Buffer& previous_layer_weights, cl::Buffer& previous_layer_synapses_output, size_t neural_network_count, size_t layer_size, size_t previous_layer_size);
cl_int opencl_feed_forward_sum_synapses(OpenCL_Data& opencl, cl::Kernel& kernel, cl::Buffer& previous_layer_synapses_output, cl::Buffer& layer_biases, cl::Buffer& layer_output, size_t previous_layer_size, size_t layer_size, size_t neural_network_count);
cl_int opencl_calculate_errors(OpenCL_Data& opencl, cl::Kernel& kernel, cl::Buffer& output_layer, cl::Buffer& errors, size_t neural_network_count, size_t output_layer_size, cl_float measured_value);

OpenCLImpl::Layer::Layer(std::uniform_real_distribution<>& distr, std::mt19937& gen, cl::Context& context, size_t neurons_count, size_t outputs_count, size_t neural_network_count) :
	single_neural_network_neurons_count(neurons_count),
	neurons_count(neurons_count * neural_network_count),
	biases(neurons_count * neural_network_count), 
	weights(neurons_count * outputs_count * neural_network_count),
	synapses_output(neurons_count * outputs_count * neural_network_count),
	output(neurons_count * neural_network_count)
{
	for (size_t k = 0; k < neural_network_count; k++) {
		for (size_t j = 0; j < neurons_count; j++) {
			for (size_t i = 0; i < outputs_count; i++) {
				weights[k * (neurons_count * outputs_count) + (j * outputs_count + i)] = static_cast<float>(distr(gen));
			}
		}
	}

	for (size_t i = 0; i < biases.size(); i++) {
		biases[i] = static_cast<float>(distr(gen));
	}

	cl_int error = CL_SUCCESS;
	biases_buffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * biases.size(), biases.data(), &error);
	weights_buffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * weights.size(), weights.data(), &error);
	synapses_output_buffer = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * synapses_output.size(), synapses_output.data(), &error);
	output_buffer = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * output.size(), output.data(), &error);
}


OpenCLImpl::MultipleNeuralNetworks::MultipleNeuralNetworks(OpenCL_Data& data, const std::vector<size_t>& topology, size_t neural_networks_count)
	: neural_networks_count(neural_networks_count), errors(neural_networks_count), data(data), helper_error_vector(neural_networks_count) {

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> distr(-1.0, 1.0);

	cl_int error;
	size_t layers_count = topology.size();
	for (size_t i = 0; i < topology.size(); i++) {
		size_t outputs_count = i == layers_count - 1 ? 0 : topology[i + 1];

		layers.push_back(Layer(distr, gen, data.context, topology[i], outputs_count, neural_networks_count));
	}
	feed_forward_kernel = cl::Kernel(data.program, "FeedForward");
	sum_kernel = cl::Kernel(data.program, "FeedForwardSum");
	errors_kernel = cl::Kernel(data.program, "CalculateErrors");
	errors_buffer = cl::Buffer(data.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * helper_error_vector.size(), helper_error_vector.data(), &error);
}


std::vector<double> OpenCLImpl::MultipleNeuralNetworks::get_errors() {
	size_t nnid = 0;
	double lowest_mean = DBL_MAX;
	size_t lowest_index = nnid;
	for (auto const& l : errors) {
		double sum = 0;
		for (auto const& r : l) {
			sum += r;
		}

		double relative_mean = sum / l.size(); 
		if (relative_mean < lowest_mean) {
			lowest_mean = relative_mean;
			lowest_index = nnid;
		}
		nnid++;
	}

	return std::vector<double>(errors[lowest_index].begin(), errors[lowest_index].end());
}


void OpenCLImpl::MultipleNeuralNetworks::feed_forward(const std::vector<double>& input, double measured_value, OpenCL_Data& opencl) {
	std::vector<cl_float>& first_layer_output = layers[0].output;
	size_t first_layer_neurons_count = layers[0].single_neural_network_neurons_count;

	if (input.size() != first_layer_neurons_count) {
		perror("Invalid input size\n");
		return;
	}

	for (size_t nn_index = 0; nn_index < neural_networks_count; nn_index++) {
		for (size_t i = 0; i < input.size(); i++) {
			first_layer_output[nn_index * first_layer_neurons_count + i] = static_cast<cl_float>(input[i]);
		}
	}

	cl_int error = CL_SUCCESS;
	layers[0].output_buffer = cl::Buffer(data.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * layers[0].output.size(), layers[0].output.data(), &error);

	for (size_t i = 0; i < layers.size() - 1; i++) {
		Layer& prev_layer = layers[i];
		Layer& next_layer = layers[i + 1];
		opencl_feed_forward(opencl,
			feed_forward_kernel,
			prev_layer.output_buffer,
			prev_layer.weights_buffer,
			prev_layer.synapses_output_buffer,
			neural_networks_count,
			next_layer.single_neural_network_neurons_count,
			prev_layer.single_neural_network_neurons_count
		);

		opencl_feed_forward_sum_synapses(opencl,
			sum_kernel,
			prev_layer.synapses_output_buffer,
			next_layer.biases_buffer,
			next_layer.output_buffer,
			prev_layer.single_neural_network_neurons_count,
			next_layer.single_neural_network_neurons_count,
			neural_networks_count
		);
	}

	Layer& output_layer = layers.back();
	opencl_calculate_errors(opencl, errors_kernel, output_layer.output_buffer, errors_buffer, neural_networks_count, output_layer.single_neural_network_neurons_count, static_cast<cl_float>(measured_value));
	data.queue.enqueueReadBuffer(errors_buffer, CL_FALSE, 0, sizeof(cl_float) * helper_error_vector.size(), helper_error_vector.data());

	for (size_t i = 0; i < helper_error_vector.size(); i++) {
		errors[i].push_back(helper_error_vector[i]);
	}
}


cl_int opencl_feed_forward(OpenCL_Data& opencl, cl::Kernel& kernel, cl::Buffer& previous_layer, cl::Buffer& previous_layer_weights, cl::Buffer& previous_layer_synapses_output,
	size_t neural_network_count, size_t layer_size, size_t previous_layer_size) {
	cl::CommandQueue& queue = opencl.queue;

	kernel.setArg(0, previous_layer);
	kernel.setArg(1, previous_layer_weights);
	kernel.setArg(2, previous_layer_synapses_output);
	kernel.setArg(3, static_cast<cl_int>(previous_layer_size));
	kernel.setArg(4, static_cast<cl_int>(layer_size));

	cl_int error = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(previous_layer_size, layer_size, neural_network_count));

	return error;
}


cl_int opencl_feed_forward_sum_synapses(OpenCL_Data& opencl, cl::Kernel& kernel, cl::Buffer& previous_layer_synapses_output, cl::Buffer& layer_biases, cl::Buffer& layer_output,
	size_t previous_layer_size, size_t layer_size, size_t neural_network_count) {
	cl::CommandQueue& queue = opencl.queue;

	kernel.setArg(0, previous_layer_synapses_output);
	kernel.setArg(1, layer_output);
	kernel.setArg(2, layer_biases);
	kernel.setArg(3, static_cast<cl_int>(layer_size));
	kernel.setArg(4, static_cast<cl_int>(previous_layer_size));

	cl_int error = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(layer_size, neural_network_count));

	return error;
}



cl_int opencl_calculate_errors(OpenCL_Data& opencl, cl::Kernel& kernel, cl::Buffer& output_layer, cl::Buffer& errors, size_t neural_network_count, size_t output_layer_size, cl_float measured_value) {
	cl::CommandQueue& queue = opencl.queue;

	kernel.setArg(0, output_layer);
	kernel.setArg(1, errors);
	kernel.setArg(2, static_cast<cl_int>(output_layer_size));
	kernel.setArg(3, measured_value);

	cl_int error = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(neural_network_count));

	return error;
}
