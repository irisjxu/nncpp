#pragma once

using namespace std;
namespace fs = std::filesystem;

enum class ColIOType { Input, Output, Unsused };

class DataColumn
{
public:
	string ColName;
	ColIOType colIoType;
	bool IsCategorical;
	list<string> CatNames; //For categorical field only
	int ColIdx;
	int featureIdx; //The first feature idx for categorical field
	float minVal = 0; //Numeric value type only
	float maxVal = 0; //Numeric value type only
};


class Dataset
{
public:
	enum class DataSetType { Training, Validation, Testing };

	Dataset() {}

	void OpenCsvDataFile(string filePath
		, const vector<bool>& vecIsCatCol
		, const vector <ColIOType>& vecColIOType
		, DataSetAlloc& dsAlloc
		, bool ignoreFirstRow /*= true*/
		, const vector<string>& colNames /*= vector<string>()*/);

	//void AddDataSrc(DataSrc ds) {}
	void CompleteLoadDataSrc();

	void UniformStandardize(float uniformMin = 0, float uniformMax = 1);
	void UniformDestandardize(vector<vector<float>>& vecOutput);
	//void NormalStandardize(float mean=0,float stddev=0){}

	const vector<DataColumn>& GetDataCols() { return dataCols; }
	void GetInputFeatures(vector<string>& features);
	void GetOutputFeatures(vector<string>& features);
	int GetInputFeatureNum() { return inputFeatureNum; }
	int GetOutputFeatureNum() { return outputFeatureNum; }

	void ResetTrainingEpoch();
	void ResetValidationRcds() { idxNextValidItem = 0; }
	void ResetTestRcds() { idxNextTestItem = 0; }

	bool RetrieveNextTrainingRcd(float** pVecInputData, float** pVecOutputData);
	bool RetrieveNextValidRcd(float** pVecInputData, float** pVecOutputData);
	bool RetrieveNextTestRcd(float** pVecInputData);

	void RetrieveTestTarget(vector<vector<float>>& vecOutputData);

	int GetTotalTrainingSamples();


protected:
	vector<DataColumn> dataCols;
	vector<int> GetNumericColIndx();

	int inputFeatureNum = 0;
	int outputFeatureNum = 0;

	int recNum = 0;
	vector <int> vecRandomSeq;
	int idxNextTrainItem = 0;
	int idxNextValidItem = 0;
	int idxNextTestItem = 0;

	vector<unique_ptr<vector<float>> > vecTrainRcdPtrs;
	vector<unique_ptr<vector<float>> > vecValidRcdPtrs;
	vector<unique_ptr<vector<float>> > vecTestRcdPtrs;

	vector<pair<float, float>> vecColMinMax;
};





