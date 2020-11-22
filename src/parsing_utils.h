#pragma once

#include <vector>
#include <string>

class Input_Parser {
public:
	Input_Parser(int& argc, char** argv);
	const std::string& get_cmd_option(const std::string& option) const;
	bool cmd_option_exists(const std::string& option) const;
	const std::string& get_arg(size_t index);
	size_t arguments_count();

private:
	std::vector<std::string> tokens;
};
