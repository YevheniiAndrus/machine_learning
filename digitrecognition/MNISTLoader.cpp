#include "MNISTLoader.h"
#include <fstream>
#include <sstream>
#include <iostream>

bool 
MNISTLoader::loadTrainingData(const std::string& fileName,
	Eigen::MatrixXd& training_data,
	Eigen::VectorXi& training_labels) {
	// allocate memory for training and expected data
	training_data.resize(training_set_size, number_pixels_in_image);
	training_labels.resize(training_set_size);

	return load(fileName, training_data, training_labels);
}

bool
MNISTLoader::loadTestData(const std::string& fileName,
	Eigen::MatrixXd& test_data,
	Eigen::VectorXi& test_labels) {
	// allocate memory for test and expected data
	test_data.resize(test_set_size, number_pixels_in_image);
	test_labels.resize(test_set_size);

	return load(fileName, test_data, test_labels);
}

bool
MNISTLoader::load(const std::string& fileName, Eigen::MatrixXd& training_data,
	Eigen::VectorXi& expected_data) {
	std::ifstream input(fileName, std::ios::in);
	if (!input.is_open()) return false;

	// read and parse data
	std::string line;
	int line_number = 0;
	while (std::getline(input, line)) {
		std::istringstream tokens(line);
		std::string token;

		// first token (number) is for expected digit
		std::getline(tokens, token, ',');
		expected_data(line_number) = std::atoi(token.c_str());

		// other tokens represents pixels in image
		std::vector<int> pixels;
		int pixel_number = 0;
		while (std::getline(tokens, token, ',')) {
			training_data(line_number, pixel_number) = std::atof(token.c_str());
			pixel_number++;
		}

		line_number++;
		std::cout << ((double)line_number / training_data.rows()) * 100 << '\r';
	}

	return true;
}