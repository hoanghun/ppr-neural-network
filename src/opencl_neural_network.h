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
		Layer(size_t neurons_count, size_t outputs_count, size_t neural_networks_count);
		size_t neurons_count;
		size_t single_neural_network_neurons_count;
		std::vector<cl_double> biases;
		std::vector<cl_double> weights;
		std::vector<cl_double> synapses_output;
		std::vector<cl_double> output;
		std::vector<cl_double> xai_accumulator;
		std::vector<cl_double> accumulator;
	private:
		static float random_weight() { return rand() / float(RAND_MAX); }
	};



	class MultipleNeuralNetworks {
	public:
		MultipleNeuralNetworks(std::vector<size_t>& topology, size_t neural_networks_count);
		void feed_forward(const std::vector<double>& input, double measured_value, opencldata& opencl);
	private:
		size_t neural_networks_count;
		void accumulate_xai(size_t neural_network_id);
		std::vector<Layer> layers;
		std::vector<std::vector<cl_double>> errors;
	};
}
