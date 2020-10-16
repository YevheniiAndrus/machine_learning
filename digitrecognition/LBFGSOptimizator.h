#pragma once

#include "LBFGSHelper.h"

class LBFGSOptimizator {
public:
	LBFGSOptimizator() = default;
	LBFGSOptimizator(double epsilon, int num_iterations);

	void set_epsilon(double epsilon) { m_epsilon = epsilon; }
	void set_numIterations(int num_iterations) { m_num_iterations = num_iterations; }

	void run(const Eigen::MatrixXd& training_matrix,
		const Eigen::VectorXi& labels);

	Eigen::VectorXd getTheta() const { return m_theta; }

private:
	Eigen::VectorXd m_theta;

	double m_epsilon;
	int m_num_iterations;
};