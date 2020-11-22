#include <CL/cl.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <algorithm>
#include <chrono>
#include <execution>

#include "nnetwork.h"
#include "sqlite/sqlite3.h"
#include "opencl_neural_network.h"

struct Measured_Value {
	std::time_t timestamp = {};
	double ist = 0;
};

static constexpr double Low_Threshold = 3.0;            //mmol/L below which a medical attention is needed
static constexpr double High_Threshold = 13.0;          //dtto above
static constexpr size_t Internal_Bound_Count = 30;      //number of bounds inside the thresholds

static constexpr double Band_Size = (High_Threshold - Low_Threshold) / static_cast<double>(Internal_Bound_Count);                        //must imply relative error <= 10%
static constexpr double Inv_Band_Size = 1.0 / Band_Size;        //abs(Low_Threshold-Band_Size)/Low_Threshold
static constexpr double Half_Band_Size = 0.5 / Inv_Band_Size;
static constexpr size_t Band_Count = Internal_Bound_Count + 2;
static constexpr size_t Output_Layer_Count = 32;
static constexpr size_t Inner_Layer_Count = 8;

double band_index_to_level(const size_t index) {
	if (index == 0) return Low_Threshold - Half_Band_Size;
	if (index >= Band_Count - 1) return High_Threshold + Half_Band_Size;

	return Low_Threshold + static_cast<double>(index - 1) * Band_Size + Half_Band_Size;
}

size_t band_level_to_index(double expected_value) {
	if (expected_value <= Low_Threshold) {
		return 0;
	}

	if (expected_value >= High_Threshold) {
		return Output_Layer_Count - 1;
	}

	double interval_ceiling = Low_Threshold;

	for (size_t i = 1; i <= Internal_Bound_Count; i++) {
		interval_ceiling += Band_Size;

		if (interval_ceiling > expected_value) {
			return i;
		}
	}

	return Output_Layer_Count - 1;
}

double risk(const double bg) {
	// DOI:  10.1080/10273660008833060
	const double original_risk = 1.794 * (pow(log(bg), 1.026) - 1.861);    //mmol/L
	return original_risk / 3.5;
}

static int callback(void* data, int argc, char** argv, char** azColName) {
	int i;
	std::vector<Measured_Value>* vector = static_cast<std::vector<Measured_Value>*>(data);
	int y, M, d, h, m;
	float s;
	Measured_Value value;
	for (i = 0; i < argc; i++) {
		if (strcmp(azColName[i], "measuredat") == 0) {
			std::tm timestamp = {};
			int rc = sscanf_s(argv[i], "%d-%d-%dT%d:%d:%f+02:00", &y, &M, &d, &h, &m, &s);
			timestamp.tm_year = y - 1900;
			timestamp.tm_mday = d;
			timestamp.tm_mon = M - 1;
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


bool load_db_data(const char* dbname, std::vector<Measured_Value>* vector) {
	sqlite3* db;
	const char* query = "SELECT * from measuredvalue";
	char* zErrMsg = 0;

	int rc = sqlite3_open(dbname, &db);

	if (rc) {
		fprintf(stderr, "Can't open database: %s.\n", sqlite3_errmsg(db));
		return false;
	}

	rc = sqlite3_exec(db, query, callback, (void*)vector, &zErrMsg);

	if (rc != SQLITE_OK) {
		fprintf(stderr, "SQL error: %s.\n", zErrMsg);
		sqlite3_free(zErrMsg);
		return false;
	}
	sqlite3_close(db);

	return true;
}

struct Results {
	std::vector<double> relative_errors;
	double mean = 0.0;
	double std_dev = 0.0;
};

struct Training_Input {
	std::vector<double> input;
	double measured_value = 0;
};

void create_training_set(const std::vector<Measured_Value>& measured_values, double minutes_prediction, std::vector<Training_Input>& training_set) {
	std::vector<Measured_Value> input_measured_values(Inner_Layer_Count);
	std::vector<double> neural_input(Inner_Layer_Count);
	Measured_Value default_value = {};

	// initialization
	size_t j = 1;
	input_measured_values[0] = measured_values[0];

	for (size_t i = 1; i < measured_values.size(); i++) {
		if (std::difftime(measured_values[i].timestamp, input_measured_values[j - 1].timestamp) == 300) {
			input_measured_values[j] = measured_values[i];
			j++;
		}
		else {
			input_measured_values[0] = measured_values[i];
			j = 1;
		}

		if (j >= Inner_Layer_Count) {
			size_t k = i + 1;

			if (k == measured_values.size()) {
				break;
			}

			double y = -1;
			while (k != measured_values.size()) {
				double diff = std::difftime(measured_values[k].timestamp, input_measured_values[Inner_Layer_Count - 1].timestamp);
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
				Training_Input single_input;
				single_input.measured_value = y;

				for (size_t index = 0; index < input_measured_values.size(); index++) {
					single_input.input.push_back(risk(input_measured_values[index].ist));
				}

				training_set.push_back(single_input);
			}

			// removing first by shifting the vector by one to the left
			j--;
			for (size_t ind = 0; ind < Inner_Layer_Count - 1; ind++) {
				input_measured_values[ind] = input_measured_values[ind + 1];
			}
			input_measured_values[Inner_Layer_Count - 1] = default_value;
		}
	}
}

void train_single_network(Neural_Network& neural_network, Results& results_struct, const std::vector<Training_Input>& training_set) {
	std::vector<double> relative_errors;
	std::vector<double> results;
	double relative_errors_sum = 0;

	// initialization

	for (auto const& sample : training_set) {
		neural_network.feed_forward(sample.input);
		neural_network.get_results(results);

		size_t output_index = 0;
		for (size_t index = 1; index < results.size(); index++) {
			if (results[index] > results[output_index]) {
				output_index = index;
			}
		}

		double x = band_index_to_level(output_index);
		double relative_error = std::abs(x - sample.measured_value) / sample.measured_value;
		relative_errors_sum += relative_error;

		std::vector<double> expected_result_output(Output_Layer_Count, 0.0);
		size_t index = band_level_to_index(sample.measured_value);

		expected_result_output[index] = 1;
		relative_errors.push_back(relative_error);
		neural_network.add_xai_intensity(relative_error);
		//neural_network.back_propagation(expected_result_output);
	}


	if (relative_errors.size() == 0) {
		printf("No training done.\n");
		return;
	}

	double mean = relative_errors_sum / relative_errors.size();

	results_struct.mean = mean;
	results_struct.relative_errors = relative_errors;
}


void process_relative_errors(std::vector<double> relative_errors) {
	double sum = std::accumulate(relative_errors.begin(), relative_errors.end(), 0.0);
	double mean = sum / relative_errors.size();

	std::sort(relative_errors.begin(), relative_errors.end());

	double std_dev_sum = 0;
	for (size_t i = 0; i < relative_errors.size(); i++) {
		std_dev_sum += (relative_errors[i] - mean) * (relative_errors[i] - mean);
	}

	double std_dev = std::sqrt(std_dev_sum / relative_errors.size());

	size_t step = relative_errors.size() / 100;

	printf("Mean of relative errors is %f and standard deviation is %f\n", mean, std_dev);
	printf("Printing cummulation function:\n");
	for (size_t i = 0; i < relative_errors.size() - 1; i += step) {
		printf("%f ", relative_errors[i]);
	}
	printf("%f\n", relative_errors.back());
}

void run_pstl_version(const std::vector<size_t>& topology, const std::vector<Training_Input>& training_set, std::vector<std::pair<Neural_Network, Results>> training) {
	std::cout << std::endl << std::endl << "Running PSTL algorithm with multiclass clasification for " << training.size() << " neural networks." << std::endl;
	std::cout << "================================================================================" << std::endl;

	auto start = std::chrono::high_resolution_clock::now();
	std::for_each(std::execution::par, training.begin(), training.end(),
		[&training_set](std::pair<Neural_Network, Results>& pair) {
			train_single_network(pair.first, pair.second, training_set);
		}
	);
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "Duration for Pstl in milliseconds: " << duration.count() << std::endl;

	auto lowest_mean = std::min_element(std::begin(training), std::end(training), [](const std::pair<Neural_Network, Results> &lhs, const std::pair<Neural_Network, Results> &rhs) {
		return lhs.second.mean < rhs.second.mean;
		}
	);

	auto &results = (*lowest_mean).second;
	auto mean = results.mean;
	auto &relative_errors = results.relative_errors;
	process_relative_errors(relative_errors);

	const Neural_Network &neural_network = (*lowest_mean).first;
	
	neural_network.export_to_svg();
	std::ofstream file("neural.ini");
	if (file.is_open()) {
		neural_network.print_neural_network(file);
	}
	file.close();
}


void run_opencl_version(const std::vector<size_t>& topology, const std::vector<Training_Input>& training_set, size_t neural_network_count) {
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	auto &platform = platforms.front();
	std::vector<cl::Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
	if (devices.size() == 0) {
		perror("No GPU");
		return;
	}
	cl::Device gpu = devices.front();

	std::ifstream helloWorldFile("super.cl");
	std::string src(std::istreambuf_iterator<char>(helloWorldFile), (std::istreambuf_iterator<char>()));
	cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));
	cl::Context context(gpu);
	cl::Program program(context, sources);

	auto err = program.build("-cl-std=CL1.2");
	OpenCL_Data data(context, program, gpu);

	OpenCLImpl::MultipleNeuralNetworks networks(data, topology, neural_network_count);

	size_t counter = 0;
	Neural_Network nn(topology);

	std::cout << std::endl << std::endl << "Running OpenCL algorithm with multiclass clasification for " << neural_network_count << " neural networks." << std::endl;
	std::cout << "================================================================================" << std::endl;

	auto start = std::chrono::high_resolution_clock::now();
	for (auto const& sample : training_set) {
		networks.feed_forward(sample.input, sample.measured_value, data);
	}

	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "Duration for OpenCL in milliseconds: " << duration.count() << std::endl << std::endl;

	auto relative_errors = networks.get_errors();
	process_relative_errors(relative_errors);
}

int main() {
	std::vector<Measured_Value> measured_values;
	if (!load_db_data("C:\\Users\\hungi\\Downloads\\asc2018.sqlite", &measured_values)) {
		exit(-1);
	}

	double minutes_prediction = 30;
	std::vector<Training_Input> training_set;
	create_training_set(measured_values, minutes_prediction, training_set);
	std::cout << "Created training size: " << training_set.size() << " using risk function." << std::endl;

	std::vector<size_t> topology{ 8, 16, 26, 32 };
	size_t training_count = 100;

	std::vector<std::pair<Neural_Network, Results>> training;
	for (size_t i = 0; i < training_count; i++) {
		Neural_Network nn(topology);
		Results results;
		training.push_back(std::make_pair(nn, results));
	}

	run_opencl_version(topology, training_set, training_count);
	run_pstl_version(topology, training_set, training);
}
