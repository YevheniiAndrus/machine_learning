#pragma once

#include <string>
#include <vector>

#include <Eigen>

class MNISTLoader {
public:
	static constexpr Eigen::Index training_set_size = 60000;
	static constexpr Eigen::Index test_set_size = 10000;

	static constexpr Eigen::Index number_pixels_in_image = 28 * 28;

	MNISTLoader() = default;

	static bool loadTrainingData(const std::string& fileName,
		Eigen::MatrixXd& training_data,
		Eigen::VectorXi& training_labels);

	static bool loadTestData(const std::string& fileName,
		Eigen::MatrixXd& test_data,
		Eigen::VectorXi& test_labels);

private:
	static bool load(const std::string& fileName, Eigen::MatrixXd& training_data,
		Eigen::VectorXi& expected_data);
};