#include "pch.h"
#include "NeuralNetworkLib.h"

using namespace Eigen;

float LossFunction::CalculateLossScoreForward(const VectorXf& predicted, const VectorXf& target)
{
	switch (lossFuncType)
	{
	case LossFunctionTypes::MSE:
		return  CalculateMSEForward(predicted, target);
	case LossFunctionTypes::CrossEntropyWithSoftmax:
		return  CalculateCrossEntropyWithSoftmaxForward(predicted, target);
	case LossFunctionTypes::CrossEntropyWithSigmoid:
		return CalculateCrossEntropyWithSigmoidForward(predicted, target);
	}
	return (float)0;
}
VectorXf LossFunction::CalculateLossScoreBackward(const VectorXf& predicted, const VectorXf& target)
{
	switch (lossFuncType)
	{
	case LossFunctionTypes::MSE:
		return  CalculateMSEBackward(predicted, target);
	case LossFunctionTypes::CrossEntropyWithSoftmax:
		return  CalculateCrossEntropyWithSoftmaxBackward(predicted, target);
	case LossFunctionTypes::CrossEntropyWithSigmoid:
		return CalculateCrossEntropyWithSigmoidBackward(predicted, target);
	}
	return VectorXf();
}

float LossFunction::CalculateMSEForward(const VectorXf& predicted, const VectorXf& target)
{
	return (target - predicted).squaredNorm() / predicted.size();
}
VectorXf LossFunction::CalculateMSEBackward(const VectorXf& predicted, const VectorXf& target)
{
	return (predicted - target) * 2 / predicted.size();
}

float LossFunction::CalculateCrossEntropyWithSoftmaxForward(const VectorXf& predicted, const VectorXf& target)
{
	return -predicted.size() * target.dot((predicted.array().log()).matrix());
}
VectorXf LossFunction::CalculateCrossEntropyWithSoftmaxBackward(const VectorXf& predicted, const VectorXf& target)
{
	return predicted - target;
}

float LossFunction::CalculateCrossEntropyWithSigmoidForward(const VectorXf& predicted, const VectorXf& target)
{
	//return (-1.0 / predicted.size()) * (target.array() * (predicted.array().log()) + (1 - target.array()) * ((1 - predicted.array()).log())).sum();
	return - (target.array() * (predicted.array().log()) + (1 - target.array()) * ((1 - predicted.array()).log())).sum();
}
VectorXf LossFunction::CalculateCrossEntropyWithSigmoidBackward(const VectorXf& predicted, const VectorXf& target)
{
	return predicted - target;
}