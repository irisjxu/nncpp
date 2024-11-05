#include "pch.h"
#include "NeuralNetworkLib.h"

void CsvDataSrc::OpenCsvDataFile(string filePath
	, const vector<bool>& vecIsCatCol
	, vector<unique_ptr<vector<float>> >& vecTrainRcdPtrs
	, vector<unique_ptr<vector<float>> >& vecValidRcdPtrs
	, vector<unique_ptr<vector<float>> >& vecTestRcdPtrs
	, const vector <ColIOType>& vecColIOType
	, DataSetAlloc& dsAlloc
	, vector<pair<float, float>>& vecColMinMax
	, bool ignoreFirstRow  /* = false*/
	, const vector<string>& colNames /*= list<string>()*/)
{
	vector<unordered_set<string>> vecCatColVal = csvFile.AnalyzeCsvFile(filePath, vecIsCatCol, ignoreFirstRow);
	recNum = csvFile.GetRowNum();
	if (ignoreFirstRow)
		recNum--;

	vecColMinMax.resize(vecIsCatCol.size(), pair(numeric_limits<float>::max(), numeric_limits<float>::min()));

	//Set column information, including column names, and categorical column unique values.
	unique_ptr<vector<string>> pOneLine;
	//Add column and set name
	vector<string> _colNames;
	if (colNames.size() > 0) //User specified column names
		_colNames = colNames;
	else if (ignoreFirstRow) // Use first row as column names
	{
		pOneLine = csvFile.ReadOneLine();
		for (auto col : *pOneLine)
			_colNames.push_back(col);
	}
	else //No column name. Name columns as "Col1", "Col2" etc
	{
		for (int i = 0; i < csvFile.GetRowNum(); i++)
			_colNames.push_back("Col" + std::to_string(i + 1));
	}

	int featureIdx = 0;
	for (int i = 0; i < vecIsCatCol.size(); i++)
	{
		DataColumn dc;
		dc.ColName = _colNames[i];
		dc.ColIdx = i;
		dc.featureIdx = featureIdx;
		dc.colIoType = vecColIOType[i];
		if (dc.colIoType != ColIOType::Unsused)
		{
			dc.IsCategorical = vecIsCatCol[i];

			if (dc.IsCategorical)
				for (const auto& val : vecCatColVal[i])
				{
					dc.CatNames.push_back(val);
					featureIdx++;
					if (dc.colIoType == ColIOType::Input)
						inputFeatureNum++;
					else
						outputFeatureNum++;
				}
			else
			{
				featureIdx++;
				if (dc.colIoType == ColIOType::Input)
					inputFeatureNum++;
				else
					outputFeatureNum++;
			}
		}
		dataCols.push_back(dc);
	}

	//Read in records
	unique_ptr < vector<string>> pCols = csvFile.ReadOneLine();

	while (pCols != nullptr)
	{
		int inp = 0;
		int outp = inputFeatureNum;
		unique_ptr<vector<float>> pRcd = make_unique< vector<float>>(inputFeatureNum + outputFeatureNum);
		for (int i = 0; i < pCols->size(); i++)
		{
			if (dataCols[i].colIoType == ColIOType::Unsused)
				continue;

			if (dataCols[i].IsCategorical) //categorical
			{
				for (auto cn : dataCols[i].CatNames)
				{
					if ((*pCols)[i] == cn)
						(*pRcd)[(dataCols[i].colIoType == ColIOType::Input ? inp++ : outp++)] = (float)1;
					else
						(*pRcd)[(dataCols[i].colIoType == ColIOType::Input ? inp++ : outp++)] = (float)0;
				}
			}
			else
			{
				if (dataCols[i].colIoType == ColIOType::Output)
					int tta = 0;
				float val = stof((*pCols)[i]);
				(*pRcd)[dataCols[i].colIoType == ColIOType::Input ? inp++ : outp++] = val;

				vecColMinMax[i].first = min(vecColMinMax[i].first, val);
				vecColMinMax[i].second = max(vecColMinMax[i].second, val);
			}
		}

		switch (dsAlloc.GetNext(pCols.get()))
		{
		case Dataset::DataSetType::Training:
			vecTrainRcdPtrs.push_back(move(pRcd));
			break;
		case Dataset::DataSetType::Validation:
			vecValidRcdPtrs.push_back(move(pRcd));
			break;
		case Dataset::DataSetType::Testing:
			vecTestRcdPtrs.push_back(move(pRcd));
			break;
		}
		pCols = csvFile.ReadOneLine();
	}

	for (int i = 0; i < dataCols.size(); i++)
	{
		if (!dataCols[i].IsCategorical)
		{
			dataCols[i].minVal = vecColMinMax[i].first;
			dataCols[i].maxVal = vecColMinMax[i].second;
		}
		else
		{
			dataCols[i].minVal = numeric_limits<float>::max();
			dataCols[i].maxVal = numeric_limits<float>::min();
		}
	}
}
