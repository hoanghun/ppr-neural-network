#pragma once

#include <vector>
#include <string>

class Input_Parser {
public:
	// Takes input and saves them as token, excluding first argument
	Input_Parser(int& argc, char** argv);

	// Returns option value for given option flag. If not present return empty string
	const std::string& get_cmd_option(const std::string& option) const;

	// Checks if the option flag is present in tokens
	bool cmd_option_exists(const std::string& option) const;

	// Returns argument on given position, if the position exceeds tokens limit returns empty string
	const std::string& get_arg(size_t index);
	
	size_t arguments_count();

private:
	std::vector<std::string> tokens;
};
