#pragma once

using namespace Eigen;
class Layer
{
public:
	int numNodes = 0;
	ActivationFunction* pActivationFunction;
	Eigen::MatrixXf weights;
	Eigen::VectorXf biases;
	Eigen::VectorXf logit;
	Eigen::VectorXf input;

	Eigen::MatrixXf weightGradientSum;
	Eigen::VectorXf biasGradientSum;

	Layer(int nN, int numPrevNodes, ActivationFunction* pAf, InitializationFunction* pWeightIf, InitializationFunction* pBiasIf);

	void Forward(const Eigen::Ref<const Eigen::VectorXf>& input, VectorXf& output);
	void Backward(const Eigen::Ref<const Eigen::VectorXf>& inputGradient, const Eigen::MatrixXf* prevWeights, VectorXf& output);
	void ForwardCalculate(const Eigen::Ref<const Eigen::VectorXf>& input, VectorXf& output);
};

class OutputLayer :public Layer
{
public:
	OutputLayer(int nN, int numPrevNodes, ActivationFunction* pAf, InitializationFunction* pWeightIf, InitializationFunction* pBiasIf)
		:Layer(nN, numPrevNodes, pAf, pWeightIf, pBiasIf) {}
	void Backward(const Eigen::Ref<const Eigen::VectorXf>& inputGradient, VectorXf& output);
};

