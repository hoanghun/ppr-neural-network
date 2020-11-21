#pragma once

#include <vector>
#include <CL/cl.hpp>

struct opencldata {
	cl::Context& context;
	cl::Program& program;
	cl::CommandQueue queue;

	opencldata(cl::Context& context, cl::Program& program, cl::Device& device) : context(context), program(program), queue(context, device) {}
};

namespace OpenCLImpl {
	class Layer {
	public:
		Layer(cl::Context& context, size_t neurons_count, size_t outputs_count, size_t neural_networks_count);
		size_t neurons_count;
		size_t single_neural_network_neurons_count;
		cl::Buffer biases_buffer;
		cl::Buffer weights_buffer;
		cl::Buffer synapses_output_buffer;
		cl::Buffer output_buffer;
		std::vector<cl_double> biases;
		std::vector<cl_double> weights;
		std::vector<cl_double> synapses_output;
		std::vector<cl_double> output;
	private:
		static float random_weight() { return rand() / float(RAND_MAX); }
	};



	class MultipleNeuralNetworks {
	public:
		MultipleNeuralNetworks(opencldata& opencl, const std::vector<size_t>& topology, size_t neural_networks_count);
		void feed_forward(const std::vector<double>& input, double measured_value, opencldata& opencl);
		std::vector<double> get_errors();
	private:
		size_t neural_networks_count;
		opencldata& data;
		cl::Kernel feed_forward_kernel;
		cl::Kernel sum_kernel;
		cl::Kernel errors_kernel;
		std::vector<Layer> layers;
		std::vector<std::vector<cl_double>> errors;
		std::vector<cl_double> helper_error_vector;
		cl::Buffer errors_buffer;
	};
}
