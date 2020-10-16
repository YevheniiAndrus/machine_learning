#pragma once

#include <Eigen>
#include "gradientdescent.h"
#include "LBFGSOptimizator.h"

class Solution {
public:
	Solution() = default;
	bool loadData(const std::string& training_data_file_name,
		const std::string& test_data_file_name);

	void logisticRegression();
	void testAccuracy();

private:
	void normalize();

	Eigen::MatrixXd adjust_data_matrix(const Eigen::MatrixXd& matrix);
	Eigen::VectorXi makeBinaryLabels(const Eigen::VectorXi& labels, int value);

private:
	Eigen::MatrixXd m_training_data;
	Eigen::VectorXi m_training_labels;

	Eigen::MatrixXd m_test_data;
	Eigen::VectorXi m_test_labels;

	std::vector<Eigen::VectorXd> m_classifiers_theta;

private:
	GradientDescent m_gd;
	LBFGSOptimizator m_lbfgs;
};