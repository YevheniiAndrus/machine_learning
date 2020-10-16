#pragma once

#include <Eigen>

class GradientDescent {
public:
	GradientDescent() = default;

	void run(const Eigen::MatrixXd& training_matrix, const Eigen::VectorXi& labels,
		int num_iterations);

	Eigen::VectorXd getTheta() const { return m_theta; }

private:
	double costFunctionLogisticRegression(const Eigen::MatrixXd& training_data,
		const Eigen::VectorXd& theta, const Eigen::VectorXi& expected_data);

	Eigen::VectorXd gradient(const Eigen::MatrixXd& training_data,
		const Eigen::VectorXd& theta, const Eigen::VectorXi& expected_data);

private:
	Eigen::VectorXd m_theta;
};