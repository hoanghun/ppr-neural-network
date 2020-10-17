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
	std::string measuredate;
	double ist = 0;
} typedef measuredvalue_t;


double count = 0.0;
double running_avg = std::numeric_limits<double>::quiet_NaN();

double Update_Average(const double level) {
	if (!std::isnan(level)) {
		count += 1.0;
		if (!std::isnan(running_avg)) {
			const double delta = level - running_avg;
			const double delta_n = delta / count;
			running_avg += delta_n;
		}
		else {
			running_avg = level;
		}

	}

	return running_avg;
}


double risk(const double bg) {
	// DOI:  10.1080/10273660008833060
	const double original_risk = 1.794 * (pow(log(bg), 1.026) - 1.861);    //mmol/L
	return original_risk / 3.5;
}

void try_iteration(double input[], int input_size, int layer_size, double layer_biases[], double layer_weights[], double layer_output[]);

static int callback(void* data, int argc, char** argv, char** azColName) {
	int i;
	std::vector<measuredvalue_t> * vector = static_cast<std::vector<measuredvalue_t> *>(data);
	measuredvalue_t value;
	for (i = 0; i < argc; i++) {
		if (strcmp(azColName[i], "measuredat") == 0) {
			value.measuredate = argv[i];
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
	const char* query = "SELECT * from measuredvalue";
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

	const int input_layer_size = 2;
	const int hidden_layer1_size = 3;
	const int hidden_layer2_size = 3;
	const int output_layer_size = 3;

	double input[8] = {};
	std::fill_n(input, 8, 1);

	double hidden_layer1_biases[hidden_layer1_size] = {};
	std::fill_n(hidden_layer1_biases, hidden_layer1_size, 1);

	double hidden_layer2_biases[hidden_layer2_size] = {};
	std::fill_n(hidden_layer2_biases, hidden_layer2_size, 1);

	double output_layer_biases[output_layer_size] = {};
	std::fill_n(output_layer_biases, output_layer_size, 1);

	double hidden_layer1_weights[input_layer_size * hidden_layer1_size] = {};
	std::fill_n(hidden_layer1_weights, input_layer_size * hidden_layer1_size, 1);

	double hidden_layer2_weights[hidden_layer1_size * hidden_layer2_size] = {};
	std::fill_n(hidden_layer2_weights, hidden_layer1_size * hidden_layer2_size, 1);

	double output_layer_weights[hidden_layer2_size * output_layer_size] = {};
	std::fill_n(output_layer_weights, hidden_layer2_size * output_layer_size, 1);

	double output[hidden_layer1_size] = {};

	try_iteration(input, input_layer_size, hidden_layer1_size, hidden_layer1_biases, hidden_layer1_weights, output);

	double output2[hidden_layer2_size] = {};
	try_iteration(output, hidden_layer1_size, hidden_layer2_size, hidden_layer2_biases, hidden_layer2_weights, output2);
}


void try_iteration(double input[], int input_size, int layer_size, double layer_biases[], double layer_weights[], double layer_output[]) {
	for (int i = 0; i < layer_size; i++) {
		double sum = 0;
		for (int j = 0; j < input_size; j++) {
			auto index = j + i * input_size;
			auto output_of_node = input[j] * layer_weights[index] + layer_biases[i];
			sum += output_of_node;

		}

		layer_output[i] = sum;
	}

	for (int i = 0; i < layer_size; i++) {
		std::cout << "output of node i: " << i << " - " << layer_output[i] << std::endl;
	}
}
