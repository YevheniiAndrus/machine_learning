#include "solution.h"
#include "MNISTLoader.h"
#include "LBFGSHelper.h"
#include <LBFGS.h>
#include <iostream>

#include "mathutil.h"

Eigen::MatrixXd
Solution::adjust_data_matrix(const Eigen::MatrixXd& matrix) {
	// add column of ones in front of matrix (most left column)
	Eigen::MatrixXd adjusted_matrix(matrix.rows(), matrix.cols() + 1);

	// column of ones
	Eigen::VectorXd ones = Eigen::VectorXd::Ones(matrix.rows());
	adjusted_matrix << ones, matrix;

	return std::move(adjusted_matrix);
}

bool
Solution::loadData(const std::string& training_data_file_name,
	const std::string& test_data_file_name) {

	std::cout << "Load training data..." << std::endl;
	if (!MNISTLoader::loadTrainingData(training_data_file_name, m_training_data,
		m_training_labels)) return false;

	std::cout << "Load test data..." << std::endl;
	if (!MNISTLoader::loadTestData(test_data_file_name, m_test_data,
		m_test_labels)) return false;

	return true;
}

Eigen::VectorXi
Solution::makeBinaryLabels(const Eigen::VectorXi& labels,
	int value) {
	
	Eigen::VectorXi output(labels.rows());
	for (Eigen::Index i = 0; i < labels.rows(); ++i) {
		if (labels[i] == value) output[i] = 1;
		else output[i] = 0;
	}

	return std::move(output);
}

void
Solution::logisticRegression() {
	normalize();

	// in order to vectorize implementation add column of ones
	// as the most left column to training data matrix
	Eigen::MatrixXd training_matrix = adjust_data_matrix(m_training_data);

	// configure multiple clussifiers using one-vs-all method
	for (int classifier = 0; classifier < 10; classifier++) {
		std::cout << "\nclassifier " << classifier << std::endl;
		
		Eigen::VectorXi binary_labels = makeBinaryLabels(m_training_labels, classifier);
		m_gd.run(training_matrix, binary_labels, 10000);

		//m_lbfgs.set_epsilon(1e-6);
		//m_lbfgs.set_numIterations(100);
		//m_lbfgs.run(training_matrix, binary_labels);

		m_classifiers_theta.push_back(m_lbfgs.getTheta());
	}
}

void Solution::normalize() {
	for (Eigen::Index c = 0; c < m_training_data.cols(); ++c) {
		m_training_data.col(c).normalize();
	}

	for (Eigen::Index c = 0; c < m_test_data.cols(); ++c) {
		m_test_data.col(c).normalize();
	}
}

void
Solution::testAccuracy()
{
	// testing
	Eigen::MatrixXd test_matrix = adjust_data_matrix(m_test_data);

	int hits = 0;
	for (int r = 0; r < test_matrix.rows(); ++r) {
		std::cout << "Test dataset #: " << r;
		double max_val = -1.0;
		int max_index = -1;
		for (int i = 0; i < m_classifiers_theta.size(); ++i) {
			// find hx with maximum value
			// index will indicate number
			double val = test_matrix.row(r) * m_classifiers_theta[i];
			double sigm_val = MathUtil::sigmoid(val);
			if (sigm_val > max_val) {
				max_val = sigm_val;
				max_index = i;
			}
		}

		if (max_index == m_test_labels[r]) {
			std::cout << "; hit" << std::endl;
			hits++;
		}
		else {
			std::cout << "; miss" << std::endl;
		}
	}

	std::cout << "Accuracy = " << ((double)hits / 10000) * 100 << "%" << std::endl;
}