#pragma once
#include "LossFunction.h"

class ModelParameters
{
public:
	int numEpochs;
	float learningRate;
	int batchSize;
	LossFunction lossFunction;

	ModelParameters(int epochs, float lr, int batch, LossFunction lossFunction);

};

