// PPR.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <algorithm>
#include <codeanalysis\warnings.h>

#include "nnetwork.h"
#pragma warning( push )
#pragma warning ( disable : ALL_CODE_ANALYSIS_WARNINGS )
#include "sqlite/sqlite3.h"
#pragma warning( pop )

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
	std::time_t timestamp = {};
	double ist = 0;
} typedef measuredvalue_t;


double risk(const double bg) {
	// DOI:  10.1080/10273660008833060
	const double original_risk = 1.794 * (pow(log(bg), 1.026) - 1.861);    //mmol/L
	return original_risk / 3.5;
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
			timestamp.tm_year = y - 1900;
			timestamp.tm_mday = d;
			timestamp.tm_mon = M;
			timestamp.tm_min = m;
			timestamp.tm_hour = h;
			timestamp.tm_sec = (int)s;
			value.timestamp = std::mktime(&timestamp);
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

struct Results {
	std::vector<double> relative_errors;
	double mean = 0.0;
	double std_dev = 0.0;
};

 void train_single_network(Neural_Network &neural_network, Results &results_struct, const std::vector<measuredvalue_t>& measured_values, double minutes_prediction) {
	std::vector<double> relative_errors;
	std::vector<double> results;
	
	size_t j = 0;
	std::vector<measuredvalue_t> input_measured_values(8);
	std::vector<double> neural_input(8);
	for (size_t i = 0; i < measured_values.size(); i++) {
		if (j == 0) {
			input_measured_values[0] = measured_values[i];
			j++;
			continue;
		}

		if (std::difftime(input_measured_values[j - 1].timestamp, measured_values[i].timestamp) == -300) {
			input_measured_values[j] = measured_values[i];
			j++;
		}
		else {
			input_measured_values[0] = measured_values[i];
			j = 1;
		}
		
		if (j == 8) {
			j = 0;
			size_t k = i + 1;


			if (k == measured_values.size()) break;
			double y = -1;
			while (k != measured_values.size()) {
				double diff = std::difftime(measured_values[k].timestamp, input_measured_values[7].timestamp);

				if (diff == minutes_prediction * 60) {
					y = measured_values[k].ist;
					break;
				}
				else if (diff > minutes_prediction * 60) {
					break;
				}
				k++;
			}

			if (y > 0) {
				for (size_t index = 0; index < input_measured_values.size(); index++) {
					neural_input[index] = risk(input_measured_values[index].ist);
				}

				neural_network.feed_forward(neural_input);
				neural_network.get_results(results);

				size_t output_index = 0;
				for (size_t index = 1; index < results.size(); index++) {
					if (results[index] > results[output_index]) {
						output_index = index;
					}
				}

				double x = Band_Index_To_Level(output_index);
				double relative_error = std::abs(x - y) / y;

				relative_errors.push_back(relative_error);
				neural_network.add_xai_intensity(relative_error);
			}
		}
	}

	std::sort(relative_errors.begin(), relative_errors.end());

	double sum = 0;
	
	for (size_t i = 0; i < relative_errors.size(); i++) {
		sum += relative_errors[i];
	}

	double mean = sum / relative_errors.size();

	double std_dev_sum = 0;
	for (size_t i = 0; i < relative_errors.size(); i++) {
		std_dev_sum += (relative_errors[i] - mean) * (relative_errors[i] - mean);
	}

	double std_dev = std_dev_sum / relative_errors.size();

	size_t step = relative_errors.size() / 100;

	printf("Mean of relative errors is %f and standard deviation is %f\n", mean, std_dev);

	size_t i = 0;
	printf("Printing cummulation function:\n");

	for (i; i < relative_errors.size() - 1; i += step) {
		printf("%f ", relative_errors[i]);
	}

	results_struct.mean = mean;
	results_struct.std_dev = std_dev;
	results_struct.relative_errors = relative_errors;

	printf("%f\n", relative_errors.back());
	printf("Done.\n");	
}

int main() {
	std::vector<measuredvalue_t> measured_values;
	load_db_data("C:\\Users\\hungi\\Downloads\\asc2018.sqlite", &measured_values);
	printf("Initializing srand\n");
	srand(static_cast<unsigned int>(time(NULL)));

	double minutes_prediction = 30;
	std::vector<unsigned> topology{ 8, 16, 26, 32 };

	
	std::vector<std::pair<Neural_Network, Results>> training;
	for (size_t i = 0; i < 10; i++) {
		Neural_Network nn(topology);
		Results results;
		training.push_back(std::make_pair(nn, results));
	}

	for (size_t i = 0; i < training.size(); i++) {
		train_single_network(training[i].first, training[i].second, measured_values, minutes_prediction);
	}

	printf("cus");
	//Neural_Network& neural_network = values.first;

	//std::ifstream infile("neuronka.txt");
	//if (infile.is_open()) {
	//	neural_network.load_weights(infile);
	//}
	//infile.close();

	//std::ofstream file("neuronka_output.txt");
	//if (file.is_open()) {
	//	neural_network.print_neural_network(file);
	//}
	//file.close();

	//neural_network.print_neural_network(std::cout);
}
