// PPR.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include "sqlite/sqlite3.h"

static constexpr double Low_Threshold = 3.0;            //mmol/L below which a medical attention is needed
static constexpr double High_Threshold = 13.0;          //dtto above
static constexpr size_t Internal_Bound_Count = 32;      //number of bounds inside the thresholds

static constexpr double Band_Size = (High_Threshold - Low_Threshold) / static_cast<double>(Internal_Bound_Count);                        //must imply relative error <= 10%
static constexpr double Inv_Band_Size = 1.0 / Band_Size;        //abs(Low_Threshold-Band_Size)/Low_Threshold
static constexpr double Half_Band_Size = 0.5 / Inv_Band_Size;
static constexpr size_t Band_Count = Internal_Bound_Count + 2;

double Band_Index_To_Level(const size_t index) {
	if (index == 0) return Low_Threshold - Half_Band_Size;
	if (index >= Band_Count - 1) return High_Threshold + Half_Band_Size;

	return Low_Threshold + static_cast<double>(index - 1) * Band_Size + Half_Band_Size;
}

struct measuredvalue {
	std::tm timestamp = {};
	double ist = 0;
} typedef measuredvalue_t;


double risk(const double bg) {
	// DOI:  10.1080/10273660008833060
	const double original_risk = 1.794 * (pow(log(bg), 1.026) - 1.861);    //mmol/L
	return original_risk / 3.5;
}


struct Connection {
	double weight;
	double delta;
};

class Neuron;
typedef std::vector<Neuron> Layer;

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

void Neuron::feed_forward(const Layer& previous_layer) {
	double sum = 0.0;

	for (int i = 0; i < previous_layer.size(); i++) {
		sum += previous_layer[i].get_output_signal() * previous_layer[i].weights[index].weight;
	}

	output_signal = sum;//Neuron::activate(sum);
}

Neuron::Neuron(unsigned outputs_count, int index) {
	for (unsigned i = 0; i < outputs_count; i++) {
		weights.push_back(Connection());
		weights.back().weight = 1; //randomWeight();
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

	double sum = 0.0;
	Layer& output_layer = layers.back();
	softmax(output_layer);
	// on output layer we'll do softmax
}

Neural_Network::Neural_Network(const std::vector<unsigned>& topology) {
	size_t number_of_layers = topology.size();
	
	for (size_t i = 0; i < number_of_layers; i++) {
		layers.push_back(Layer());
		unsigned outputs_count = i == number_of_layers - 1 ? 0 : topology[i + 1];

		for (unsigned j = 0; j <= topology[i]; j++) { // <= because of bias node
			layers.back().push_back(Neuron(outputs_count, j));
			printf("Made a neuron with index %d\n", j);
		}

		layers.back().back().set_output_signal(1.0); // bias node
	}
}


static int callback(void* data, int argc, char** argv, char** azColName) {
	int i;
	std::vector<measuredvalue_t> * vector = static_cast<std::vector<measuredvalue_t> *>(data);
	int y, M, d, h, m;
	float s;
	measuredvalue_t value;
	for (i = 0; i < argc; i++) {
		if (strcmp(azColName[i], "measuredat") == 0) {
			std::tm timestamp = {};
			int rc = sscanf_s(argv[i], "%d-%d-%dT%d:%d:%f+02:00", &y, &M, &d, &h, &m, &s);
			timestamp.tm_year = y;
			timestamp.tm_mday = d;
			timestamp.tm_mon = M;
			timestamp.tm_min = m;
			timestamp.tm_hour = h;
			timestamp.tm_sec = (int)s;
			value.timestamp = timestamp;
		}
		if (strcmp(azColName[i], "ist") == 0) {
			value.ist = atof(argv[i]);
		}
		
	}
	vector->push_back(value);

	return 0;
}


bool load_db_data(const char* dbname, std::vector<measuredvalue_t>* vector) {
	sqlite3* db;
	const char* query = "SELECT * from measuredvalue order by measuredat";
	char* zErrMsg = 0;
	const char* data = "Callback function called";

	int rc = sqlite3_open(dbname, &db);
	printf("Query: %s\n", query);

	if (rc) {
		fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(db));
		return false;
	}
	else {
		fprintf(stderr, "Opened database successfully\n");
	}

	rc = sqlite3_exec(db, query, callback, (void*)vector, &zErrMsg);

	if (rc != SQLITE_OK) {
		fprintf(stderr, "SQL error: %s\n", zErrMsg);
		sqlite3_free(zErrMsg);
	}
	else {
		fprintf(stdout, "Operation done successfully\n");
	}
	sqlite3_close(db);

	return true;
}

int main() {
	std::vector<measuredvalue_t> vector;
	load_db_data("C:\\Users\\hungi\\Downloads\\asc2018.sqlite", &vector);
	std::vector<unsigned> topology;
	topology.push_back(3);
	topology.push_back(2);
	topology.push_back(1);

	Neural_Network neural_network(topology);
	std::vector<double> input_vals;
	input_vals.push_back(1);  
	input_vals.push_back(1);  
	input_vals.push_back(1);  
	neural_network.feed_forward(input_vals);
	//std::vector<double> target_vals;
	//target_vals.push_back(1);
	//target_vals.push_back(1);
	//neural_network.back_propagation(target_vals);

	std::vector<double> result_vals;
	neural_network.get_results(result_vals);
}
