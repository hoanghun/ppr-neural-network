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
	Neuron(size_t outputs_count, size_t index, std::uniform_real_distribution<>& distr, std::mt19937& gen);
	void set_output_signal(double val) { output_signal = val; }
	double get_output_signal() const { return output_signal; }
	void feed_forward(Layer& previous_layer);
	void calculate_output_gradients(double target_value);
	void calculate_hidden_gradients(const Layer& next_layer);
	void update_input_weights(Layer& previous_layer);
	void set_weight(size_t index, double weight);
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
	Neural_Network(const std::vector<size_t>& topology);
	void feed_forward(const std::vector<double>& input_vals);
	void back_propagation(const std::vector<double>& target_vals);
	void get_results(std::vector<double>& result_vals);
	void print_neural_network(std::ostream &file) const;
	void load_weights(std::ifstream& file);
	void add_xai_intensity(double error);
	void export_to_svg() const;
private:
	void get_layer_edge_intensities(const Layer& layer, double& max_intensity, double& max_xai_intensity, double& min_intensity, double& min_xai_intensity) const;
	std::vector<Layer> layers;
	double error = 0;
};
