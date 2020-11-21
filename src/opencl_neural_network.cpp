#include "opencl_neural_network.h"


cl_int opencl_feed_forward(opencldata& opencl, std::vector<cl_double>& previous_layer, std::vector<cl_double>& previous_layer_weights,
	std::vector<cl_double>& previous_layer_synapses_output, size_t neural_network_count, size_t layer_size, size_t previous_layer_size);
cl_int opencl_feed_forward_sum_synapses(opencldata& opencl, std::vector<cl_double>& previous_layer_synapses_output, std::vector<cl_double>& layer_biases, std::vector<cl_double>& layer_output,
	size_t previous_layer_size, size_t layer_size, size_t neural_network_count);
cl_int opencl_add_to_accumulators(opencldata& opencl, std::vector<cl_double>& accumulator, std::vector<cl_double>& to_add);
cl_int opencl_calculate_errors(opencldata& opencl, std::vector<cl_double>& output_layer, std::vector<cl_double>& errors, size_t neural_network_count, size_t output_layer_size, cl_double measured_value);

OpenCLImpl::Layer::Layer(size_t neurons_count, size_t outputs_count, size_t neural_network_count) :
	single_neural_network_neurons_count(neurons_count),
	neurons_count(neurons_count * neural_network_count),
	biases(neurons_count * neural_network_count), 
	weights(neurons_count * outputs_count * neural_network_count),
	synapses_output(neurons_count * outputs_count * neural_network_count),
	output(neurons_count * neural_network_count),
	xai_accumulator(neurons_count * outputs_count * neural_network_count, 0),
	accumulator(neurons_count * outputs_count * neural_network_count, 0)
{
	for (size_t k = 0; k < neural_network_count; k++) {
		for (size_t j = 0; j < neurons_count; j++) {
			for (size_t i = 0; i < outputs_count; i++) {
				weights[k * (neurons_count * outputs_count) + (j * outputs_count + i)] = random_weight();
			}
		}
	}

	for (size_t i = 0; i < biases.size(); i++) {
		biases[i] = 0;
	}
}


OpenCLImpl::MultipleNeuralNetworks::MultipleNeuralNetworks(std::vector<size_t>& topology, size_t neural_networks_count) : neural_networks_count(neural_networks_count), errors(neural_networks_count) {
	size_t layers_count = topology.size();
	for (size_t i = 0; i < topology.size(); i++) {
		size_t outputs_count = i == layers_count - 1 ? 0 : topology[i + 1];

		layers.push_back(Layer(topology[i], outputs_count, neural_networks_count));
	}
}
void OpenCLImpl::MultipleNeuralNetworks::accumulate_xai(size_t neural_network_id) {
	for (size_t i = 0; i < layers.size() - 1; i++) {
		Layer& layer = layers[i];
		size_t weights_count = (layer.weights.size() / neural_networks_count);
		size_t offset = neural_network_id * weights_count;
		
		for (size_t j = offset; j < offset + weights_count; j++) {
			layer.xai_accumulator[j] += layer.synapses_output[j];
		}
	}
}

void OpenCLImpl::MultipleNeuralNetworks::feed_forward(const std::vector<double>& input, double measured_value, opencldata& opencl) {
	std::vector<cl_double>& first_layer_output = layers[0].output;
	size_t first_layer_neurons_count = layers[0].single_neural_network_neurons_count;

	if (input.size() != first_layer_neurons_count) {
		perror("Invalid input size\n");
		return;
	}

	for (size_t nn_index = 0; nn_index < neural_networks_count; nn_index++) {
		for (size_t i = 0; i < input.size(); i++) {
			first_layer_output[nn_index * first_layer_neurons_count + i] = static_cast<cl_double>(input[i]);
		}
	}

	for (size_t i = 0; i < layers.size() - 1; i++) {
		Layer& prev_layer = layers[i];
		Layer& next_layer = layers[i + 1];
		opencl_feed_forward(opencl,
			prev_layer.output,
			prev_layer.weights,
			prev_layer.synapses_output,
			neural_networks_count,
			next_layer.single_neural_network_neurons_count,
			prev_layer.single_neural_network_neurons_count
		);

		opencl_feed_forward_sum_synapses(opencl,
			prev_layer.synapses_output,
			next_layer.biases,
			next_layer.output,
			prev_layer.single_neural_network_neurons_count,
			next_layer.single_neural_network_neurons_count,
			neural_networks_count
		);

		//opencl_add_to_accumulators(opencl, prev_layer.accumulator, prev_layer.synapses_output);
	}

	std::vector<cl_double> errors(neural_networks_count, 0);

	Layer& output_layer = layers.back();
	opencl_calculate_errors(opencl, output_layer.output, errors, neural_networks_count, output_layer.single_neural_network_neurons_count, static_cast<cl_double>(measured_value));
	//for (size_t nnid = 0; nnid < neural_networks_count; nnid++) {
	//	this->errors[nnid].push_back(errors[nnid]);
	//	if (errors[nnid] < 0.60 && nnid % 2 == 0) {
	//		accumulate_xai(nnid);
	//	}
	//}
}


cl_int opencl_feed_forward(opencldata& opencl, std::vector<cl_double>& previous_layer, std::vector<cl_double>& previous_layer_weights,
	std::vector<cl_double>& previous_layer_synapses_output, size_t neural_network_count, size_t layer_size, size_t previous_layer_size) {

	cl::Context& context = opencl.context;
	cl::Program& program = opencl.program;
	cl::CommandQueue& queue = opencl.queue;

	cl_int error;

	cl::Buffer inArray(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * previous_layer.size(), previous_layer.data(), &error);
	cl::Buffer inBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * previous_layer_weights.size(), previous_layer_weights.data(), &error);
	cl::Buffer outBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(cl_double) * previous_layer_synapses_output.size(), &error);
	cl::Kernel kernel(program, "FeedForward");

	kernel.setArg(0, inArray);
	kernel.setArg(1, inBuf);
	kernel.setArg(2, outBuf);
	kernel.setArg(3, static_cast<cl_int>(previous_layer_size));
	kernel.setArg(4, static_cast<cl_int>(layer_size));

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(previous_layer_size, layer_size, neural_network_count));
	error = queue.enqueueReadBuffer(outBuf, CL_FALSE, 0, sizeof(cl_double) * previous_layer_synapses_output.size(), previous_layer_synapses_output.data());

	return error;
}


cl_int opencl_feed_forward_sum_synapses(opencldata& opencl, std::vector<cl_double>& previous_layer_synapses_output, std::vector<cl_double>& layer_biases, std::vector<cl_double>& layer_output,
	size_t previous_layer_size, size_t layer_size, size_t neural_network_count) {

	cl_int error;
	cl::Context& context = opencl.context;
	cl::Program& program = opencl.program;
	cl::CommandQueue& queue = opencl.queue;

	cl::Buffer inBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * previous_layer_synapses_output.size(), previous_layer_synapses_output.data(), &error);
	cl::Buffer inBiases(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * layer_biases.size(), layer_biases.data(), &error);
	cl::Buffer outBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(cl_double) * layer_output .size(), &error);
	cl::Kernel kernel(program, "FeedForwardSum");

	kernel.setArg(0, inBuf);
	kernel.setArg(1, outBuf);
	kernel.setArg(2, inBiases);
	kernel.setArg(3, static_cast<cl_int>(layer_size));
	kernel.setArg(4, static_cast<cl_int>(previous_layer_size));

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(layer_size, neural_network_count));
	error = queue.enqueueReadBuffer(outBuf, CL_FALSE, 0, sizeof(cl_double) * layer_output.size(), layer_output.data());

	return error;
}


cl_int opencl_add_to_accumulators(opencldata& opencl, std::vector<cl_double>& accumulator, std::vector<cl_double>& to_add) {
	cl_int error;
	cl::Context& context = opencl.context;
	cl::Program& program = opencl.program;
	cl::CommandQueue& queue = opencl.queue;

	cl::Buffer accum_buf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * accumulator.size(), accumulator.data(), &error);
	cl::Buffer to_add_buf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * to_add.size(), to_add.data(), &error);
	cl::Buffer output_buf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(cl_double) * accumulator .size(), &error);
	cl::Kernel kernel(program, "AddTwoArrays");

	kernel.setArg(0, accum_buf);
	kernel.setArg(1, to_add_buf);
	kernel.setArg(2, output_buf);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(accumulator.size()));
	error = queue.enqueueReadBuffer(output_buf, CL_FALSE, 0, sizeof(cl_double) * accumulator.size(), accumulator.data());

	return error;
}


cl_int opencl_calculate_errors(opencldata& opencl, std::vector<cl_double>& output_layer, std::vector<cl_double>& errors, size_t neural_network_count, size_t output_layer_size, cl_double measured_value) {
	cl_int error;
	cl::Context& context = opencl.context;
	cl::Program& program = opencl.program;
	cl::CommandQueue& queue = opencl.queue;

	cl::Buffer inArray(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * output_layer.size(), output_layer.data(), &error);
	cl::Buffer outBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(cl_double) * errors.size(), &error);
	cl::Kernel kernel(program, "CalculateErrors");

	kernel.setArg(0, inArray);
	kernel.setArg(1, outBuf);
	kernel.setArg(2, static_cast<cl_int>(output_layer_size));
	kernel.setArg(3, measured_value);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(neural_network_count));
	error = queue.enqueueReadBuffer(outBuf, CL_FALSE, 0, sizeof(cl_double) * errors.size(), errors.data());

	return error;
}
