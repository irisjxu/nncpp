#include "pch.h"
#include "NeuralNetworkLib.h"

DataSetAllocRatio::DataSetAllocRatio(int trainingSetRatio, int validSetRatio, int testSetRatio)
{
	_trainingSetRatio = trainingSetRatio;
	_validSetRatio = validSetRatio;
	_testSetRatio = testSetRatio;
	Reset();
}

Dataset::DataSetType DataSetAllocRatio::GetNext(const vector<string> * pRcdData)
{
	float rTrain = FLT_MAX;
	if (_trainingSetRatio > 0)
		rTrain = (float)(1.0 * trainigRecNum / _trainingSetRatio);
	float rValid = FLT_MAX;
	if (_validSetRatio > 0)
		rValid = (float)(1.0 * validRecNum / _validSetRatio);
	float rTest = FLT_MAX;
	if (_testSetRatio > 0)
		rTest = (float)(1.0 * testRecNum / _testSetRatio);
	
	if (rTrain <= rValid && rTrain <= rTest)
	{
		trainigRecNum++;
		return Dataset::DataSetType::Training;
	}
	else if (rValid <= rTest)
	{
		validRecNum++;
		return Dataset::DataSetType::Validation;
	}
	else
	{
		testRecNum++;
		return Dataset::DataSetType::Testing;
	}
}