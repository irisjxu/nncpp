#include "pch.h"
#include "NeuralNetworkLib.h"

Layer::Layer(int nN, int numPrevNodes, ActivationFunction* pAf, InitializationFunction* pWeightIf, InitializationFunction* pBiasIf)
{
	numNodes = nN;
	pActivationFunction = pAf;

	weights.resize(nN, numPrevNodes);
	weightGradientSum.resize(nN, numPrevNodes);
	weightGradientSum.setZero();
	biases.resize(numNodes);
	biasGradientSum.resize(numNodes);
	biasGradientSum.setZero();
	input.resize(numPrevNodes);
	input.setZero();

	pWeightIf->initialize(weights);
	pBiasIf->initialize(biases);
}

void Layer::Forward(const Eigen::Ref<const Eigen::VectorXf>& inputVec, VectorXf& output)
{
	input = inputVec;
	logit = weights * this->input + biases;
	output = pActivationFunction->CalculateForward(logit);
	//cout << "I:\n" << input << "\nW:\n" << weights << "\nB:\n" << biases << "\n\n";
}

void Layer::Backward(const Eigen::Ref<const Eigen::VectorXf>& inputGradient, const Eigen::MatrixXf* prevWeights, VectorXf& output)
{
	output = ((prevWeights->transpose() * inputGradient).array() * (pActivationFunction->CalculateBackward(logit)).array()).matrix();
	biasGradientSum += output;
	weightGradientSum += output * input.transpose();
}

void OutputLayer::Backward(const Eigen::Ref<const Eigen::VectorXf>& inputGradient, VectorXf& output)
{
	//output = output * (pActivationFunction->CalculateBackward(logit).array()).matrix();
	output = (output.array() * pActivationFunction->CalculateBackward(logit).array()).matrix();
	biasGradientSum += output;
	weightGradientSum += output * input.transpose();
}

void Layer::ForwardCalculate(const Eigen::Ref<const Eigen::VectorXf>& input, VectorXf& output)
{
	output = pActivationFunction->CalculateForward(weights * input + biases);
}