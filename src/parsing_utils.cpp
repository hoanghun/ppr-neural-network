#include "parsing_utils.h"

Input_Parser::Input_Parser(int& argc, char** argv) {
	for (int i = 1; i < argc; ++i)
		this->tokens.push_back(std::string(argv[i]));
}

const std::string& Input_Parser::get_cmd_option(const std::string& option) const {
	std::vector<std::string>::const_iterator itr;
	itr = std::find(this->tokens.begin(), this->tokens.end(), option);
	if (itr != this->tokens.end() && ++itr != this->tokens.end()) {
		return *itr;
	}
	static const std::string empty_string("");
	return empty_string;
}

bool Input_Parser::cmd_option_exists(const std::string& option) const {
	return std::find(this->tokens.begin(), this->tokens.end(), option)
		!= this->tokens.end();
}

const std::string& Input_Parser::get_arg(size_t index) {
	if (index < tokens.size()) {
		return tokens[index];
	}
	else {
		static const std::string empty_string("");
		return empty_string;
	}
}

size_t Input_Parser::arguments_count() {
	return tokens.size();
}

