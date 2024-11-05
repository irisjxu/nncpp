#include "pch.h"
#include "NeuralNetworkLib.h"

void Dataset::OpenCsvDataFile(string filePath
	, const vector<bool>& vecIsCatCol
	, const vector <ColIOType>& vecColIOType
	, DataSetAlloc& dsAlloc
	, bool ignoreFirstRow /*= true*/
	, const vector<string>& colNames /*= vector<string>()*/)
{
	CsvDataSrc csvDs;

	csvDs.OpenCsvDataFile(filePath, vecIsCatCol, vecTrainRcdPtrs, vecValidRcdPtrs, vecTestRcdPtrs,
		vecColIOType, dsAlloc, vecColMinMax, ignoreFirstRow, colNames);
	dataCols = csvDs.GetDataCols();

	inputFeatureNum = csvDs.GetInputFeatureNum();
	outputFeatureNum = csvDs.GetOutputFeatureNum();
	recNum = csvDs.GetRecNum();
}
void Dataset::CompleteLoadDataSrc()
{
	//Initialize training vector randome order.
	vecRandomSeq.resize(vecTrainRcdPtrs.size());
	for (int i = 0; i < vecTrainRcdPtrs.size(); i++)
		vecRandomSeq[i] = i;

	ResetTrainingEpoch();
}

void Dataset::ResetTrainingEpoch()
{
	shuffle(vecRandomSeq.begin(), vecRandomSeq.end(), default_random_engine());
	idxNextTrainItem = 0;
}

void Dataset::GetInputFeatures(vector<string>& features)
{
	features.clear();
	for (auto it = dataCols.begin(); it != dataCols.end(); it++)
	{
		if (it->colIoType == ColIOType::Input)
		{
			if (it->IsCategorical)
			{
				for (auto it2 = it->CatNames.begin(); it2 != it->CatNames.end(); it2++)
					features.push_back(it->ColName + "_" + *it2);
			}
			else
				features.push_back(it->ColName);
		}
	}
}

void Dataset::GetOutputFeatures(vector<string>& features)
{
	features.clear();
	for (auto it = dataCols.begin(); it != dataCols.end(); it++)
	{
		if (it->colIoType == ColIOType::Output)
		{
			if (it->IsCategorical)
			{
				for (auto it2 = it->CatNames.begin(); it2 != it->CatNames.end(); it2++)
					features.push_back(it->ColName + "_" + *it2);
			}
			else
				features.push_back(it->ColName);
		}
	}
}

bool Dataset::RetrieveNextTrainingRcd(float** pVecInputData, float** pVecOutputData)
{
	if (idxNextTrainItem >= vecTrainRcdPtrs.size())
		return false;
	(*pVecInputData) = vecTrainRcdPtrs[vecRandomSeq[idxNextTrainItem]].get()->data();
	(*pVecOutputData) = (*pVecInputData) + inputFeatureNum;
	idxNextTrainItem++;
	return true;
}
bool Dataset::RetrieveNextValidRcd(float** pVecInputData, float** pVecOutputData)
{
	if (idxNextValidItem >= vecValidRcdPtrs.size())
		return false;

	(*pVecInputData) = vecValidRcdPtrs[idxNextValidItem].get()->data();
	(*pVecOutputData) = (*pVecInputData) + inputFeatureNum;
	idxNextValidItem++;
	return true;
}
bool Dataset::RetrieveNextTestRcd(float** pVecInputData)
{
	if (idxNextTestItem >= vecTestRcdPtrs.size())
		return false;
	(*pVecInputData) = vecTestRcdPtrs[idxNextTestItem].get()->data();
	idxNextTestItem++;
	return true;
}

void Dataset::RetrieveTestTarget(vector<vector<float>>& vecOutputData)
{
	for (auto& pRcd : vecTestRcdPtrs)
	{
		float* pOutStart = pRcd.get()->data() + inputFeatureNum;
		vecOutputData.push_back(vector<float>(pOutStart, pOutStart + outputFeatureNum));
	}
	UniformDestandardize(vecOutputData);
}

vector<int> Dataset::GetNumericColIndx()
{
	vector<int>rst;
	size_t colNo = 0;
	for (auto col : dataCols)
	{
		if (col.IsCategorical)
			colNo += col.CatNames.size();
		else
			rst.push_back((int)(colNo++));
	}
	return rst;
}
void Dataset::UniformStandardize(float uniformMin /*= 0*/, float uniformMax /*= 1*/)
{
	vector<int> vecNumericColIdx = GetNumericColIndx();
	for (int i = 0; i < vecTrainRcdPtrs.size(); i++)
		for (int j = 0; j < vecNumericColIdx.size(); j++)
			(*(vecTrainRcdPtrs[i]))[vecNumericColIdx[j]] = ((*(vecTrainRcdPtrs[i]))[vecNumericColIdx[j]] - vecColMinMax[vecNumericColIdx[j]].first) / (vecColMinMax[vecNumericColIdx[j]].second - vecColMinMax[vecNumericColIdx[j]].first);
	for (int i = 0; i < vecValidRcdPtrs.size(); i++)
		for (int j = 0; j < vecNumericColIdx.size(); j++)
			(*(vecValidRcdPtrs[i]))[vecNumericColIdx[j]] = ((*(vecValidRcdPtrs[i]))[vecNumericColIdx[j]] - vecColMinMax[vecNumericColIdx[j]].first) / (vecColMinMax[vecNumericColIdx[j]].second - vecColMinMax[vecNumericColIdx[j]].first);
	for (int i = 0; i < vecTestRcdPtrs.size(); i++)
		for (int j = 0; j < vecNumericColIdx.size(); j++)
			(*(vecTestRcdPtrs[i]))[vecNumericColIdx[j]] = ((*(vecTestRcdPtrs[i]))[vecNumericColIdx[j]] - vecColMinMax[vecNumericColIdx[j]].first) / (vecColMinMax[vecNumericColIdx[j]].second - vecColMinMax[vecNumericColIdx[j]].first);
}
void Dataset::UniformDestandardize(vector<vector<float>>& vecOutput)
{
	int outputColIdx = 0;
	for (int i = 0; i < dataCols.size(); i++)
	{
		if (!(dataCols[i].colIoType == ColIOType::Output)
			|| dataCols[i].IsCategorical)
			continue;

		for (vector<float>& vals : vecOutput)
			vals[dataCols[i].featureIdx - GetInputFeatureNum()] =
			vals[dataCols[i].featureIdx - GetInputFeatureNum()]
			* (dataCols[i].maxVal - dataCols[i].minVal)
			+ dataCols[i].minVal;
	}
}
int Dataset::GetTotalTrainingSamples() {
	return vecTrainRcdPtrs.size();
}