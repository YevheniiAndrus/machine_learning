#include "LBFGSOptimizator.h"
#include <LBFGS.h>

LBFGSOptimizator::LBFGSOptimizator(double epsilon,
	int num_iterations) {

	m_epsilon = epsilon;
	m_num_iterations = num_iterations;
}

void LBFGSOptimizator::run(const Eigen::MatrixXd& training_matrix,
	const Eigen::VectorXi& labels) {

	LBFGSHelper helper(training_matrix, labels);
	LBFGSpp::LBFGSParam<double> param;
	param.epsilon = 1e-6;
	param.max_iterations = 100;

	LBFGSpp::LBFGSSolver<double> solver(param);
	double fx = 0.0;

	m_theta = Eigen::VectorXd::Zero(training_matrix.cols());
	solver.minimize(helper, m_theta, fx);
}