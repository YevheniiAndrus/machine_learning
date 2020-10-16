#pragma once

#include <Eigen>

class MathUtil {
public:
	MathUtil() = default;

	static double lambda;
	static double learning_rate;

	static double sigmoid(int value) {
		return 1.0 / (1.0 + std::exp(-value));
	}

	static double sigmoid(double value) {
		return 1.0 / (1.0 + std::exp(-value));
	}

	static void sigmoid(Eigen::VectorXd& data) {
		for (Eigen::Index i = 0; i < data.rows(); ++i) {
			data[i] = sigmoid(data[i]);
		}
	}
};