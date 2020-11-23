#pragma once
#include <vector>
#include <fstream>
#include <random>

class Neuron;

typedef std::vector<Neuron> Layer;

struct Connection {
	double weight;
	double delta;
	double intensity_counter;
	double xai_intensity_counter;
	double current_intensity; // value that went through the synapse in the last run, used to calculate xai
};

class Neuron {
public:
	// Creates a single neuron with number of weight equaling to outputs_count and uses passed generator to generate random weights.
	// Index value indicates order in the layer
	Neuron(size_t outputs_count, size_t index, std::uniform_real_distribution<>& distr, std::mt19937& gen);

	void set_output_signal(double val) { output_signal = val; }
	double get_output_signal() const { return output_signal; }

	// Calculates output signal for this neuron.
	// Sums all outputs from neurons from previous layer + bias neuron.
	// After summation applies tanh function and saves the output.
	// Increases intensity counter for every weight connected to this neuron
	void feed_forward(Layer& previous_layer);

	// Backprop functions
	void calculate_output_gradients(double target_value);
	void calculate_hidden_gradients(const Layer& next_layer);
	void update_input_weights(Layer& previous_layer);

	// Sets weight on given index to given weight
	void set_weight(size_t weight_index, double weight);

	// Adds last intensity that went through synapses of this neuron to explainable artificial intelligence counter.
	// This should happen only if the error was less or equal to 0.15.
	void add_xai_intensity();

	const std::vector<Connection> &get_weights() const { return weights; }

private:
	static double eta;
	static double alpha;
	static double activate(double x);
	static double derivative(double x);
	double output_signal;
	double sumDOW(const Layer& next_layer) const;
	double gradient;
	size_t index;

	static double random_weight() { return rand() / double(RAND_MAX); }

	std::vector<Connection> weights;
};

class Neural_Network {
public:
	// Creates a neural network with given topology. Every layer has one more fake neuron called bias.
	Neural_Network(const std::vector<size_t>& topology);

	// Feed forward through entire network. Output layer than has the results.
	void feed_forward(const std::vector<double>& input_vals);

	// Back propagation algorithm to update weights.
	void back_propagation(const std::vector<double>& target_vals);

	// Fills passed vector with outputs of output layer.
	void get_results(std::vector<double>& result_vals);

	// Outputs neural network weights into given output stream.
	void print_neural_network(std::ostream &file) const;

	// Load neural network weight from given input file stream.
	void load_weights(std::ifstream& file);

	// Adds to explainable artificial counters if the error is less or equal to 0.15.
	void add_xai_intensity(double error);

	// Loads neural network template from file nn.svg. If this file is malformed output is undefined and will most likely be unusable.
	// Replaces the synapses colors with normalized values (0, 255).
	// This normalization is done on every layer to ilustrate values passed in single layers.
	void export_to_svg(std::ofstream& out_intensity_file, std::ofstream& out_xai_intensity_file) const;
private:
	void get_layer_edge_intensities(const Layer& layer, double& max_intensity, double& max_xai_intensity, double& min_intensity, double& min_xai_intensity) const;
	std::vector<Layer> layers;
	double error = 0;
};
