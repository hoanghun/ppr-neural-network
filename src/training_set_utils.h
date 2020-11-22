#include <vector>

struct Training_Input {
	std::vector<double> input;
	double measured_value = 0;
};

bool create_training_set(const std::string& db_name, size_t input_layer_count, size_t minutes_prediction, std::vector<Training_Input>& training_set);
