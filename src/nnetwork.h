#pragma once
#include <vector>
#include <fstream>

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
	Neuron(unsigned outputs_count, int index);
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
	int index;

	static double random_weight() { return rand() / double(RAND_MAX); }

	std::vector<Connection> weights;
};

class Neural_Network {
public:
	Neural_Network(const std::vector<unsigned>& topology);
	void feed_forward(const std::vector<double>& input_vals);
	void back_propagation(const std::vector<double>& target_vals);
	void get_results(std::vector<double>& result_vals);
	void print_neural_network(std::ostream &file);
	void load_weights(std::ifstream& file);
	void add_xai_intensity(double error);
private:
	std::vector<Layer> layers;
	double error = 0;
};
