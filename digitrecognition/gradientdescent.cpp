#include "gradientdescent.h"
#include "mathutil.h"
#include <iostream>

void
GradientDescent::run(const Eigen::MatrixXd& training_matrix,
	const Eigen::VectorXi& labels,
	int num_iterations) {

	m_theta = Eigen::VectorXd::Zero(training_matrix.cols());
	int iteration = 0;
	while (iteration < num_iterations) {
		Eigen::VectorXd grad = gradient(training_matrix, m_theta, labels);
		m_theta = m_theta - grad * MathUtil::learning_rate;
		double cost = costFunctionLogisticRegression(training_matrix, m_theta, labels);
		std::cout << "iteration# " << iteration++ << "; cost = " << cost << std::endl;
	}
}

double
GradientDescent::costFunctionLogisticRegression(const Eigen::MatrixXd& training_data,
	const Eigen::VectorXd& theta, const Eigen::VectorXi& expected_data) {
	// cost function for logistic regression
	// J(theta) = sum[-y(i) * log(z)) - (1 - y(i)) * log(1 - h(z))] / m
	// where h - logistic hypothes function
	// z = logistic function of x*theta
	// theta - parameters to be tuned
	// y - expected values
	// i = current set
	// run for i = (0, m), where m - number of training sets

	// in order to avoid loops we are using vectorizing implementation
	// first lets calculate vector of hypothesis
	double sum = 0.0;
	for (Eigen::Index r = 0; r < training_data.rows(); ++r) {
		double hx = training_data.row(r) * theta;
		double hx_sigm = MathUtil::sigmoid(hx);

		sum += static_cast<double>(expected_data[r]) * std::log(hx_sigm) + (1.0 - static_cast<double>(expected_data[r])) *
			std::log(1 - hx_sigm);
	}

	sum = -sum / training_data.rows();

	// regularization
	double squared_theta = theta.transpose() * theta;
	squared_theta = squared_theta * MathUtil::lambda / (2 * training_data.rows());

	sum += squared_theta;

	return sum;
}

Eigen::VectorXd 
GradientDescent::gradient(const Eigen::MatrixXd& training_data,
	const Eigen::VectorXd& theta, const Eigen::VectorXi& expected_data) {
	// calculate gradient by following rule
	// dJ/dThetaj = [sum(hx(i) - yi) * xj] / m
	// i = number of training sets
	// j = current feature for which gradient calculated
	// sum from i:num_training_set
	// output is a vector (size correspond to number of features)

	Eigen::VectorXd hx = training_data * theta;
	MathUtil::sigmoid(hx);

	Eigen::MatrixXd expected_data_d = expected_data.cast<double>();
	Eigen::VectorXd grad = training_data.transpose() * (hx - expected_data_d);
	grad = grad / training_data.rows();

	// regularize gradient
	// !!! Do not regularize first theta
	Eigen::VectorXd temp = theta;
	temp(0) = 0;
	grad += temp * (MathUtil::lambda / training_data.rows());

	return std::move(grad);
}