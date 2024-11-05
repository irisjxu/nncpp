#pragma once
class InitializationFunction
{
public:
	virtual void initialize(Eigen::MatrixXf& parameters) = 0;
	virtual void initialize(Eigen::VectorXf& parameters) = 0;
};

//initialize all values to a constant (0 by default)
class ConstantInitializationFunction : public InitializationFunction
{
public:
	float value;
	ConstantInitializationFunction() { value = 0; }
	ConstantInitializationFunction(float val) { value = val; }
	void initialize(Eigen::MatrixXf& parameters)
	{
		parameters.setConstant(value);
	}
	void initialize(Eigen::VectorXf& parameters)
	{
		parameters.setConstant(value);
	}
};

//normal with mean = 0 by default, variance = k/n -> n is the number of input nodes, k is a set constant (2 by default)
//k=2: He initializer
//k=1: Xavier initializer
class NormalDistInitializationFunction : public InitializationFunction
{
public:
	float m;
	float k;
	float maxSDAway = 2;
	NormalDistInitializationFunction() { m = 0; k = 2; }
	NormalDistInitializationFunction(float varianceConstant) { m = 0; k = varianceConstant; }
	NormalDistInitializationFunction(float mean, float varianceConstant) { m = mean; k = varianceConstant; }

	void initialize(Eigen::MatrixXf& parameters);
	void initialize(Eigen::VectorXf& parameters);

protected:
	float getNextNormalDist(size_t numRows);
	std::default_random_engine generator;
	std::normal_distribution<float> distribution; // default mean=0.0; stddev=1.0
};

