#pragma once
using namespace Eigen;

class LossFunction {
public:
	enum class LossFunctionTypes { MSE, CrossEntropyWithSoftmax, CrossEntropyWithSigmoid };
	LossFunctionTypes lossFuncType;

	float CalculateLossScoreForward(const VectorXf& predicted, const VectorXf& target);
	VectorXf CalculateLossScoreBackward(const VectorXf& predicted, const VectorXf& target);

protected:
	float CalculateMSEForward(const VectorXf& predicted, const VectorXf& target);
	VectorXf CalculateMSEBackward(const VectorXf& predicted, const VectorXf& target);

	float CalculateCrossEntropyWithSoftmaxForward(const VectorXf& predicted, const VectorXf& target);
	VectorXf CalculateCrossEntropyWithSoftmaxBackward(const VectorXf& predicted, const VectorXf& target);

	float CalculateCrossEntropyWithSigmoidForward(const VectorXf& predicted, const VectorXf& target);
	VectorXf CalculateCrossEntropyWithSigmoidBackward(const VectorXf& predicted, const VectorXf& target);
};