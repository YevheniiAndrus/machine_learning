#include "LBFGSHelper.h"
#include "mathutil.h"

LBFGSHelper::LBFGSHelper(const Eigen::MatrixXd& training_matrix,
	const Eigen::VectorXi& labels) :
	m_training_matrix(training_matrix),
	m_labels(labels)
{
}

double LBFGSHelper::operator()(const Eigen::VectorXd& theta,
	Eigen::VectorXd& grad) {

	Eigen::VectorXd hx = m_training_matrix * theta;
	MathUtil::sigmoid(hx);

	Eigen::MatrixXd expected_data_d = m_labels.cast<double>();
	grad = m_training_matrix.transpose() * (hx - expected_data_d);
	grad = grad / m_training_matrix.rows();

	// regularize gradient
	// !!! Do not regularize first theta
	Eigen::VectorXd temp = theta;
	temp(0) = 0;
	grad += temp * (1.0 / m_training_matrix.rows());

	// cost function
	double sum = 0.0;
	for (Eigen::Index r = 0; r < m_training_matrix.rows(); ++r) {
		double hx = m_training_matrix.row(r) * theta;
		double hx_sigm = MathUtil::sigmoid(hx);

		sum += static_cast<double>(m_labels[r]) * std::log(hx_sigm) + (1.0 - static_cast<double>(m_labels[r])) *
			std::log(1 - hx_sigm);
	}

	sum = -sum / m_training_matrix.rows();

	// regularization
	double squared_theta = theta.transpose() * theta;
	squared_theta = squared_theta * 1.0 / (2 * m_training_matrix.rows());

	sum += squared_theta;

	return sum;
}