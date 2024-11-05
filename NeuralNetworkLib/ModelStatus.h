#pragma once

class ModelResult
{
public:
	int epochs = 0;
	float trainLoss = 0;
	float validLoss = 0;
	float validAccuracy = 0;
};

class ModelStatus
{
public:
	bool running = false;
	list<ModelResult> listRst;
	unique_ptr<vector<vector<float>>> pTestPredictVals;
	ModelStatus() { pTestPredictVals = make_unique<vector<vector<float>>>(); }
};



