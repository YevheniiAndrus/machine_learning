#pragma once

#include <Eigen>

class LBFGSHelper {
public:
	LBFGSHelper(const Eigen::MatrixXd& training_matrix,
		const Eigen::VectorXi& labels);

	double operator()(const Eigen::VectorXd& theta, Eigen::VectorXd& grad);

private:
	const Eigen::MatrixXd& m_training_matrix;
	const Eigen::VectorXi& m_labels;
};