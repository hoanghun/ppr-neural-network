#include <vector>
#include <ctime>
#include <string>

#include "sqlite/sqlite3.h"
#include "training_set_utils.h"

struct Measured_Value {
	std::time_t timestamp = {};
	double ist = 0;
};


static int callback(void* data, int argc, char** argv, char** column_name) {
	std::vector<Measured_Value>* vector = static_cast<std::vector<Measured_Value>*>(data);
	int y, M, d, h, m;
	float s;
	Measured_Value value;
	for (size_t i = 0; i < argc; i++) {
		if (strcmp(column_name[i], "measuredat") == 0) {
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
		if (strcmp(column_name[i], "ist") == 0) {
			value.ist = atof(argv[i]);
		}

	}
	vector->push_back(value);

	return 0;
}


bool load_db_data(const char* dbname, std::vector<Measured_Value>* vector) {
	sqlite3* db = nullptr;
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

double risk(const double bg) {
	// DOI:  10.1080/10273660008833060
	const double original_risk = 1.794 * (pow(log(bg), 1.026) - 1.861);    //mmol/L
	return original_risk / 3.5;
}

bool create_training_set(const std::string& db_name, size_t input_layer_count, double minutes_prediction, std::vector<Training_Input>& training_set) {
	std::vector<Measured_Value> measured_values;
	if (!load_db_data(db_name.c_str(), &measured_values)) {
		return false;
	}

	std::vector<Measured_Value> input_measured_values(input_layer_count);
	std::vector<double> neural_input(input_layer_count);
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

		if (j >= input_layer_count) {
			size_t k = i + 1;

			if (k == measured_values.size()) {
				break;
			}

			double y = -1;
			while (k != measured_values.size()) {
				double diff = std::difftime(measured_values[k].timestamp, input_measured_values[input_layer_count - 1].timestamp);
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
			for (size_t ind = 0; ind < input_layer_count - 1; ind++) {
				input_measured_values[ind] = input_measured_values[ind + 1];
			}
			input_measured_values[input_layer_count - 1] = default_value;
		}
	}

	return training_set.size() > 0;
}
