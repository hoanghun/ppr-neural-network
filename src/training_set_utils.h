#include <vector>

struct Training_Input {
	std::vector<double> input;
	double measured_value = 0;
};


// Creates training set for given minute prediction value and database name.
// Training set is created using shifting window. Algorithm attempts to move the window by one, e.g. 1-8, 2-9.
// Returns success if the training set was created
// false otherwise
bool create_training_set(const std::string& db_name, size_t input_layer_count, size_t minutes_prediction, std::vector<Training_Input>& training_set);
