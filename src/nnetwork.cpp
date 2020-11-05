#include "nnetwork.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

void Neuron::update_input_weights(Layer& previous_layer) {
	for (int i = 0; i < previous_layer.size(); i++) {
		Neuron& current = previous_layer[i];
		double old_delta_weight = current.weights[index].delta;
		double new_delta_weight = eta * current.get_output_signal() * gradient + alpha * old_delta_weight;
		current.weights[index].delta = new_delta_weight;
		current.weights[index].weight += new_delta_weight;
	}
}
double Neuron::sumDOW(const Layer& next_layer) const {
	double sum = 0.0;

	for (int i = 0; i < next_layer.size() - 1; i++) {
		sum += weights[i].weight * next_layer[i].gradient;
	}

	return sum;
}

void Neuron::calculate_hidden_gradients(const Layer& next_layer) {
	double dow = sumDOW(next_layer);
	gradient = dow * Neuron::derivative(output_signal);
}

void Neuron::calculate_output_gradients(double target_value) {
	double delta = target_value - output_signal;
	gradient = delta * Neuron::derivative(output_signal);
}

double Neuron::activate(double x) {
	return tanh(x);
}

double Neuron::derivative(double x) {
	return 1.0 - tanh(x) * tanh(x);
}

void Neuron::feed_forward(Layer& previous_layer) {
	double sum = 0.0;

	for (int i = 0; i < previous_layer.size(); i++) {
		double current_intensity = previous_layer[i].get_output_signal() * previous_layer[i].weights[index].weight; 
		sum += current_intensity;
		previous_layer[i].weights[index].intensity_counter += current_intensity;
		previous_layer[i].weights[index].current_intensity = current_intensity;
	}

	output_signal = Neuron::activate(sum);
}

Neuron::Neuron(unsigned outputs_count, int index) {
	for (unsigned i = 0; i < outputs_count; i++) {
		weights.push_back(Connection());
		weights.back().weight = random_weight();
	}

	this->index = index;
	this->gradient = 0;
	this->output_signal = 0;
}

void softmax(std::vector<Neuron>& input) {
	double max = -INFINITY;
	size_t input_size_without_bias = input.size() - 1;
	for (size_t i = 0; i < input_size_without_bias; i++) {
		if (input[i].get_output_signal() > max) {
			max = input[i].get_output_signal();
		}
	}

	double sum = 0.0;
	for (size_t i = 0; i < input_size_without_bias; i++) {
		input[i].set_output_signal(exp(input[i].get_output_signal() - max));
		sum += input[i].get_output_signal();
	}

	for (size_t i = 0; i < input_size_without_bias; i++) {
		input[i].set_output_signal(input[i].get_output_signal() / sum);
	}
}

void Neural_Network::get_results(std::vector<double>& result_vals) {
	result_vals.clear();

	for (int i = 0; i < layers.back().size() - 1; i++) {
		result_vals.push_back(layers.back()[i].get_output_signal());
	}
}

void Neural_Network::back_propagation(const std::vector<double>& target_values) {
	Layer& output_layer = layers.back();
	// calculating overall net ERROR RMS
	error = 0.0;

	for (int i = 0; i < output_layer.size() - 1; i++) {
		double delta = target_values[i] - output_layer[i].get_output_signal();
		error += delta * delta;
	}

	error /= output_layer.size() - 1;
	error = sqrt(error);

	// calculating gradients on output layer
	for (int i = 0; i < output_layer.size() - 1; i++) {
		output_layer[i].calculate_output_gradients(target_values[i]);
	}

	// calculating gradients on hidden layers
	for (size_t layer_index = layers.size() - 2; layer_index > 0; layer_index--) {
		Layer& hidden_layer = layers[layer_index];
		size_t next_layer_index = layer_index + 1;
		Layer& next_layer = layers[next_layer_index];

		for (size_t i = 0; i < hidden_layer.size(); i++) {
			hidden_layer[i].calculate_hidden_gradients(next_layer);
		}
	}


	// updating weights
	for (size_t layer_index = layers.size() - 1; layer_index > 0; layer_index--) {
		Layer& layer = layers[layer_index];
		size_t previous_layer_index = layer_index - 1;
		Layer& previous_layer = layers[previous_layer_index];

		for (size_t i = 0; i < layer.size() - 1; i++) {
			layer[i].update_input_weights(previous_layer);
		}
	}
}


void Neural_Network::feed_forward(const std::vector<double>& input_values) {
	if (input_values.size() != layers[0].size() - 1) {
		return;
	}

	for (int i = 0; i < input_values.size(); i++) {
		layers[0][i].set_output_signal(input_values[i]);
	}

	for (int layer_index = 1; layer_index < layers.size(); layer_index++) {
		int previous_layer_index = layer_index - 1;
		Layer& previous_layer = layers[previous_layer_index];
		for (int neuron_index = 0; neuron_index < layers[layer_index].size() - 1; neuron_index++) {
			layers[layer_index][neuron_index].feed_forward(previous_layer);
		}
	}

	//Layer& output_layer = layers.back();
	//softmax(output_layer);
	// on output layer we'll do softmax
}

Neural_Network::Neural_Network(const std::vector<unsigned>& topology) {
	size_t number_of_layers = topology.size();

	for (size_t i = 0; i < number_of_layers; i++) {
		layers.push_back(Layer());
		unsigned outputs_count = i == number_of_layers - 1 ? 0 : topology[i + 1];

		for (unsigned j = 0; j <= topology[i]; j++) { // <= because of bias node
			layers.back().push_back(Neuron(outputs_count, j));
		}

		layers.back().back().set_output_signal(1.0); // bias node
	}
}
void Neuron::set_weight(size_t index, double weight) {
	if (index < weights.size()) {
		weights[index].weight = weight;
	}
	else {
		printf("Invalid index %zu\n", index);
	}
}

void Neural_Network::print_neural_network(std::ostream &file) {
	for (size_t layer_index = 0; layer_index < layers.size() - 1; layer_index++) {
		file << "[hidden_layer_" << layer_index + 1 << "]" << std::endl;

		Layer &layer = layers[layer_index];
		for (size_t neuron_index = 0; neuron_index < layer.size(); neuron_index++) {
			const std::vector<Connection>& weights = layer[neuron_index].get_weights();
			

			for (size_t weight_index = 0; weight_index < weights.size(); weight_index++) {
				file << "Neuron" << weight_index << "_Weight" << neuron_index  << "=" << weights[weight_index].weight << std::endl;
			}
		}
	}
}

void Neural_Network::load_weights(std::ifstream& file) {
	std::string line;
	size_t layer_index = -1;
	
	while (std::getline(file, line)) {
		if (line.find("[hidden_layer_") != std::string::npos) {
			layer_index++;
		}
		else {
			size_t weight_index, neuron_index;
			double weight;

			std::replace_if(line.begin(), line.end(), [](const char& c) { return c != '.' && !std::isdigit(c); }, ' ');
			std::istringstream iss(line);

			if (!(iss >> weight_index >> neuron_index >> weight)) {
				printf("");
			}

			if (layer_index < layers.size() && neuron_index < layers[layer_index].size()) {
				layers[layer_index][neuron_index].set_weight(weight_index, weight);
			}
			else {
				printf("Invalid indicies layer_index: %zu or neuron_index: %zu\n", layer_index, neuron_index);
			}
		}
	}
}

void Neuron::add_xai_intensity() {
	for (size_t weight_index = 0; weight_index < weights.size(); weight_index++) {
		weights[weight_index].xai_intensity_counter += weights[weight_index].current_intensity;
	}
}

void Neural_Network::add_xai_intensity(double error) {
	if (error <= 0.15) {
		printf("Error in the limits, adding xai.\n");
		for (size_t layer_index = 0; layer_index < layers.size(); layer_index++) {
			Layer& layer = layers[layer_index];
			for (size_t neuron_index = 0; neuron_index < layer.size(); neuron_index++) {
				layer[neuron_index].add_xai_intensity();
			}
		}
	}
	else {
		printf("Error too big, ignoring..\n");
	}
}
