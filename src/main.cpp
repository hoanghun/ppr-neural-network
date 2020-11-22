#include <CL/cl.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <execution>
#include <sstream>

#include "nnetwork.h"
#include "opencl_neural_network.h"
#include "training_set_utils.h"
#include "parsing_utils.h"

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

struct Results {
	std::vector<double> relative_errors;
	double mean = 0.0;
	double std_dev = 0.0;
};


// Trains a single neural network against training set. Results are then saved into passed struct.
// If backprop flag is true, then backprop algorithm is applied
void train_single_network(Neural_Network& neural_network, Results& results_struct, const std::vector<Training_Input>& training_set, bool backprop) {
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
		if (backprop) {
			neural_network.back_propagation(expected_result_output);
		}
	}


	if (relative_errors.size() == 0) {
		printf("No training done.\n");
		return;
	}

	double mean = relative_errors_sum / relative_errors.size();

	results_struct.mean = mean;
	results_struct.relative_errors = relative_errors;
}


// Calculates relative error mean and it's standard deviation.
// If the flag save_to_csv is true, then relative errors are sampled into distribution function and saved into csv along with mean and sd.
void process_relative_errors(std::vector<double> relative_errors, bool save_to_csv) {
	double sum = std::accumulate(relative_errors.begin(), relative_errors.end(), 0.0);
	double mean = sum / relative_errors.size();

	std::sort(relative_errors.begin(), relative_errors.end());

	double std_dev_sum = 0;
	for (size_t i = 0; i < relative_errors.size(); i++) {
		std_dev_sum += (relative_errors[i] - mean) * (relative_errors[i] - mean);
	}

	double std_dev = std::sqrt(std_dev_sum / relative_errors.size());

	size_t step = relative_errors.size() / 100;

	std::cout << "Mean of relative errors is " << mean << " and standard deviation is " << std_dev << "." << std::endl << std::endl;
	if (save_to_csv) {
		std::ofstream errors_csv("errors.csv");
		if (errors_csv.is_open()) {
			errors_csv << mean << "," << std_dev << ",";
			for (size_t i = 0; i < relative_errors.size() - 1; i += step) {
				errors_csv << relative_errors[i] << " ";
			}
			errors_csv << relative_errors.back() << std::endl;

			std::cout << "Saved errors into file errors.csv." << std::endl;
		}
		else {
			std::cout << "Cannot write into file 'error.csv'" << std::endl;
		}
	}
}

// Runs PSTL version of parallel neural network implementation.
// After the run find the best neural network and saves it's weight to neural.ini file and it's errors to errors.csv.
void run_pstl_version(const std::vector<Training_Input>& training_set, std::vector<std::pair<Neural_Network, Results>> training) {
	std::cout << std::endl << std::endl << "Running PSTL algorithm with multiclass clasification for " << training.size() << " neural networks." << std::endl;
	std::cout << "================================================================================" << std::endl;

	auto start = std::chrono::high_resolution_clock::now();
	std::for_each(std::execution::par, training.begin(), training.end(),
		[&training_set](std::pair<Neural_Network, Results>& pair) {
			train_single_network(pair.first, pair.second, training_set, false);
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
	auto &relative_errors = results.relative_errors;
	process_relative_errors(relative_errors, true);

	Neural_Network &neural_network = (*lowest_mean).first;
	
	neural_network.export_to_svg();
	std::ofstream file("neural.ini");
	if (file.is_open()) {
		neural_network.print_neural_network(file);
		std::cout << "Saved neural network weights into file neural.ini." << std::endl;
	}
}



// Runs OpenCL implementation of parallel neural networks.
// There has to be valid GPU device to run on.
void run_opencl_version(const std::vector<size_t>& topology, const std::vector<Training_Input>& training_set, size_t neural_network_count) {
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	auto &platform = platforms.front();
	std::vector<cl::Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
	if (devices.size() == 0) {
		std::cout << "No GPU" << std::endl;
		return;
	}
	cl::Device gpu = devices.front();

	std::ifstream openCL_file("../src/neural_network.cl");
	if (!openCL_file.is_open()) {
		std::cout << "Invalid open cl file" << std::endl;
		return;
	}
	std::string src(std::istreambuf_iterator<char>(openCL_file), (std::istreambuf_iterator<char>()));
	cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));
	cl::Context context(gpu);
	cl::Program program(context, sources);

	auto err = program.build("-cl-std=CL1.2");
	if (err != CL_SUCCESS) {
		std::cout << "Didnt build the opencl program" << std::endl;
		return;
	}
	OpenCL_Data data(context, program, gpu);

	OpenCLImpl::MultipleNeuralNetworks networks(data, topology, neural_network_count);

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
	process_relative_errors(relative_errors, false);
}


// Loads weights from a file and creates a neural network with these weights.
// This neural network is then run against the training set.
void load_neural_network(const std::string& weights_file_name, const std::vector<size_t>& topology, const std::vector<Training_Input>& training_set) {
	Neural_Network nn(topology);
	Results results;
	std::ifstream file(weights_file_name);
	if (file.is_open()) {
		std::cout << std::endl << std::endl << "Loaded weights from  " << weights_file_name << ", running against the training set." << std::endl;
		std::cout << "================================================================================" << std::endl;
		nn.load_weights(file);

		train_single_network(nn, results, training_set, false);
		process_relative_errors(results.relative_errors, false);
	}
	else {
		std::cout << "Invalid weights file name." << std::endl;
	}

}

void print_help() {
	std::cout << "Parallel programming semestral work by Hung Ngoc Hoang." << std::endl;
	std::cout << "Usage: ./PPR MINUTES_PREDICTION SQLITE_DB_PATH [options]" << std::endl;
	std::cout << "ARGS:" << std::endl;
	std::cout << "\t<MINUTES_PREDICTION>\tnumber of minutes for prediction." << std::endl;
	std::cout << "\t<SQLITE_DB_PATH>\tpath to database with ist values and dates." << std::endl;
	std::cout << "OPTIONS:" << std::endl;
	std::cout << "\t-s <NUM>\tset <NUM> of instances to train." << std::endl;
	std::cout << "\t-t <WEIGHTS>\tload weights from <WEIGHTS> file path and run single neural network." << std::endl;
	std::cout << "\t-opencl\t\trun the opencl version." << std::endl;
	std::cout << "\t-pstl\t\trun the pstl version." << std::endl;
	std::cout << "\t-both\t\trun opencl and pstl version." << std::endl;
}

int main(int argc, char* argv[]) {
	Input_Parser input_parser(argc, argv);
	std::vector<Training_Input> training_set;
	std::vector<size_t> topology{ 8, 16, 26, 32 };
	
	if (input_parser.arguments_count() < 2) {
		std::cout << input_parser.arguments_count() << std::endl;
		print_help();
		return 0;
	}

	size_t minutes;
	std::stringstream sstream(input_parser.get_arg(0));
	sstream >> minutes;
	if (sstream.fail()) {
		std::cout << "Invalid minutes prediction value." << std::endl;
		print_help();
		return 0;
	}

	std::cout << "Prediction is set to " << minutes << " minutes." << std::endl;
	const std::string& db_name = input_parser.get_arg(1);
	bool created = create_training_set(db_name, Inner_Layer_Count, minutes, training_set);
	if (!created) {
		print_help();
		return 0;
	}
	std::cout << "Created training size: " << training_set.size() << " using risk function." << std::endl;
	bool only_single_train = false;
	if (input_parser.cmd_option_exists("-t")) {
		const std::string& training_file_name = input_parser.get_cmd_option("-t");
		load_neural_network(training_file_name, topology, training_set);
		only_single_train = true;
	}

	size_t training_count = 0;
	if (input_parser.cmd_option_exists("-s")) {
		sstream.clear();
		sstream.str(input_parser.get_cmd_option("-s"));
		sstream >> training_count;
		if (sstream.fail()) {
			std::cout << "Passed invalid number in -s flag." << std::endl;
		}
		else {
			std::cout << "Number of neural networks instances set to " << training_count << "." << std::endl;
		}
	}
	else {
		training_count = 100;
	}

	std::vector<std::pair<Neural_Network, Results>> training;
	for (size_t i = 0; i < training_count; i++) {
		Neural_Network nn(topology);
		Results results;
		training.push_back(std::make_pair(nn, results));
	}

	if (input_parser.cmd_option_exists("-pstl")) {
		run_pstl_version(training_set, training);
		only_single_train = true;
	}

	if (input_parser.cmd_option_exists("-opencl")) {
		run_opencl_version(topology, training_set, training_count);
		only_single_train = true;
	}

	if (!only_single_train || input_parser.cmd_option_exists("-both")) {
		run_opencl_version(topology, training_set, training_count);
		run_pstl_version(training_set, training);
	}
}
