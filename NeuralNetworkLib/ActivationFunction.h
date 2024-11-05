#pragma once
using namespace std;
using namespace Eigen;

class ActivationFunction
{
public:
	virtual VectorXf CalculateForward(VectorXf x) = 0;
	virtual VectorXf CalculateBackward(VectorXf x) = 0;
};

class LinearActivationFunction : public ActivationFunction
{
	VectorXf CalculateForward(VectorXf x) { return x; }
	VectorXf CalculateBackward(VectorXf x) { return  VectorXf::Constant(x.size(), (float)1); }
};

class SigmoidWithCrossEntropyActivationFunction : public ActivationFunction
{
	VectorXf CalculateForward(VectorXf x)
	{
		return x.unaryExpr([](float x) -> float {return 1 / (1 + exp(-x)); });
	}
	VectorXf CalculateBackward(VectorXf x)
	{
		return x;
	}
};

class ReluActivationFunction : public ActivationFunction
{
protected:
	/*
	float slopeLeft = 0;
	float xTurnPoint = 0;
	float yTurnPoint = 0;
	float slopeRight = 1;
	*/
public:
	VectorXf CalculateForward(VectorXf x)
	{
		return x.unaryExpr([](float x) -> float {return max((float)0, x); });
	}
	VectorXf CalculateBackward(VectorXf x)
	{
		return x.unaryExpr([](float x) -> float {return (x > 0 ? (float)1 : (float)0); });
	}
};

class SoftmaxWithCrossEntropyActivationFunction : public ActivationFunction
{
protected:
public:
	VectorXf CalculateForward(VectorXf x)
	{
		return x.array().exp() / (x.array().exp().sum());
	}
	VectorXf CalculateBackward(VectorXf x)
	{
		return x;
	}
};







