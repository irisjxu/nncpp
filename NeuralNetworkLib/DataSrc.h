#pragma once

class DataSrc
{
protected:
	vector< DataColumn> dataCols;
	int recNum = 0;
public:
	const int GetRecNum() { return recNum; }
};

class CsvDataSrc : public DataSrc
{
protected:
	CsvLib csvFile;
	void Close() { csvFile.CloseFile(); }

	int inputFeatureNum = 0;
	int outputFeatureNum = 0;

public:
	int GetInputFeatureNum() { return inputFeatureNum; }
	int GetOutputFeatureNum() { return outputFeatureNum; }

	CsvDataSrc() { recNum = 0; }
	const vector< DataColumn> GetDataCols() { return dataCols; }
	//If ignoreFirstRow==true and colNames==null, then use first row as column names.
	void OpenCsvDataFile(string filePath
		, const vector<bool>& vecIsCatCol
		, vector<unique_ptr<vector<float>> > & vecTrainRcdPtrs
		, vector<unique_ptr<vector<float>> > & vecValidRcdPtrs
		, vector<unique_ptr<vector<float>> > & vecTestRcdPtrs
		, const vector <ColIOType>& vecColIOType
		, DataSetAlloc& dsAlloc
		, vector<pair<float, float>> &colMinMax
		, bool ignoreFirstRow = true
		, const vector<string>& colNames = vector<string>());

	void AddCsvDataDataFile(string folderPath, DataSetAlloc& dsAlloc, bool ignoreFirstRow = true) {}
};

