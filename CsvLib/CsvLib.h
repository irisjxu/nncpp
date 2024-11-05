
#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_set>
using namespace std;

class CsvLib
{
protected:
	string filePath="";
	ifstream fileStream;
	int rowNum =0;
	int colNum = 0;	//based on row 1
protected:
	unique_ptr<vector<string>> AnalysizeCsvLine(const string& rowLine); //return the fields (string) in vector.

public:
	CsvLib() { rowNum = 0; colNum = 0; }
	~CsvLib() { CloseFile(); }
	vector<unordered_set<string>> AnalyzeCsvFile(string setFilePath, const vector<bool>& isCatCol, bool ignoreFirstRow);
	int GetRowNum() { return rowNum; }
	unique_ptr<vector<string>> ReadOneLine();
	void CloseFile();


};