
#pragma once
class DataSetAlloc
{
public:
	virtual Dataset::DataSetType GetNext(const vector<string> * pRcdData) =0;
};


class DataSetAllocRatio : public DataSetAlloc
{
protected:
	int _trainingSetRatio;
	int _validSetRatio;
	int _testSetRatio;

	int trainigRecNum;
	int validRecNum;
	int testRecNum;
public:
	DataSetAllocRatio(int trainingSetRatio = 1, int validSetRatio = 0, int TestSetRatio = 0);
	void Reset() { trainigRecNum = 0; validRecNum = 0; testRecNum = 0; }


	Dataset::DataSetType GetNext(const vector<string>* pRcdData = nullptr);
};

class DataSetAllocCustomized : public DataSetAlloc
{
	DataSetAllocCustomized(/*function to determine dataset type*/) {}
	Dataset::DataSetType GetNext(const vector<string>& RcdData) {}
};