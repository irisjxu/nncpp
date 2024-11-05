#include "pch.h"
#include "NeuralNetworkLib.h"
#include "InitializationFunction.h"

float NormalDistInitializationFunction::getNextNormalDist(size_t numPrevNodes)
{
	float r;
	while (true)
	{
		r = distribution(generator);
		if (r<maxSDAway && r>-maxSDAway)
			break;
	}
	r = m + r * sqrt(k / numPrevNodes);
	return r;
}

void NormalDistInitializationFunction::initialize(Eigen::MatrixXf& parameters)
{
	for (int i = 0; i < parameters.rows(); i++)
		for (int j = 0; j < parameters.cols(); j++)
			parameters(i, j) = getNextNormalDist(parameters.rows());
}

void NormalDistInitializationFunction::initialize(Eigen::VectorXf& parameters)
{
	for (int i = 0; i < parameters.rows(); i++)
			parameters(i) = getNextNormalDist(parameters.rows());
}