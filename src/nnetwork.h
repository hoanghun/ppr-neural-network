#pragma once
#include <vector>

class Neuron;

typedef std::vector<Neuron> Layer;

struct Connection {
	double weight;
	double delta;
};

class Neuron {
public:
	Neuron(unsigned outputs_count, int index);
	void set_output_signal(double val) { output_signal = val; }
	double get_output_signal() const { return output_signal; }
	void feed_forward(const Layer& previous_layer);
	void calculate_output_gradients(double target_value);
	void calculate_hidden_gradients(const Layer& next_layer);
	void update_input_weights(Layer& previous_layer);

private:
	static double eta;
	static double alpha;
	static double activate(double x);
	static double derivative(double x);
	double output_signal;
	double sumDOW(const Layer& next_layer) const;
	double gradient;
	int index;

	static double randomWeight() { return rand() / double(RAND_MAX); }

	std::vector<Connection> weights;
};

class Neural_Network {
public:
	Neural_Network(const std::vector<unsigned>& topology);
	void feed_forward(const std::vector<double>& input_vals);
	void back_propagation(const std::vector<double>& target_vals);
	void get_results(std::vector<double>& result_vals);
private:
	std::vector<Layer> layers;
	double error = 0;
};
