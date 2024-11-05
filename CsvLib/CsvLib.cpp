#include "CsvLib.h"


// TODO: This is an example of a library function

vector<unordered_set<string>> CsvLib::AnalyzeCsvFile(string setFilePath,const vector<bool> & vecIsCatCol, bool ignoreFirstRow)
{
	filePath = setFilePath;
	//fileStream.open(setFilePath); //default move is open for read

	fileStream.open(setFilePath);

	string line;
	rowNum = 0;
	int tRowNum = 0;

	unique_ptr < vector<string>> pCols;
	vector<unordered_set<string>> vecUniqueValues(vecIsCatCol.size(), unordered_set<string>());

	if (getline(fileStream, line))
	{
		pCols = AnalysizeCsvLine(line);
		if (pCols->size() == 0)
			throw new runtime_error ("csv file wrong format at first row");
		colNum = (int)(pCols->size());
		if (colNum!= vecIsCatCol.size())
			throw new runtime_error("Number of column doesn't match isCatCol vector size");
		tRowNum++;
	}
	if (!ignoreFirstRow)
	{
		for (int i=0;i< vecIsCatCol.size();i++)
		{
			if (vecIsCatCol[i])
				vecUniqueValues[i].insert((*pCols)[i]);
		}
	}

	while (getline(fileStream, line))
	{
		pCols = AnalysizeCsvLine(line);
		if (pCols->size() == 0)
			throw new runtime_error("csv file wrong format at row" + tRowNum);
		tRowNum++;

		for (int i = 0; i < vecIsCatCol.size(); i++)
		{
			if (vecIsCatCol[i])
				vecUniqueValues[i].insert((*pCols)[i]);
		}
	}

	rowNum = tRowNum;

	fileStream.clear();
	fileStream.seekg(0);

	return vecUniqueValues;
}

unique_ptr<vector<string>> CsvLib::ReadOneLine()
{
	string line;

	unique_ptr < vector<string>> pCols = nullptr;

	if (getline(fileStream, line))
		pCols = AnalysizeCsvLine(line);
	return pCols;
}

void CsvLib::CloseFile()
{
	rowNum = 0;
	colNum = 0;
	filePath = "";

	if (fileStream.is_open())
		fileStream.close();
}

unique_ptr<vector<string>> CsvLib::AnalysizeCsvLine(const string& rowLine)
{
	unique_ptr<vector<string>> pRetCol =  make_unique<vector<string>>();
	//vector<string> cols ;
	string curCol = "";
	//10: Before a field start
	//20: In a non-quotation mark enclosed field
	//30: In a quotation mark enclosed field.
	//40: Read one quotation mark inside a quotation mark enclosed field.
	//50: Finished
	//100: Error
	int state = 10; 

	string line= rowLine + '\r';
	for (int i = 0; i < line.length(); i++)
	{

		switch (state)
		{
		case 10:
			switch (line[i])
			{
			case ','://Add empty field
				pRetCol-> push_back("");
				break;
			case '"':
				state = 30;
				break;
			case '\r':
			case '\n':
				state = 50;
				pRetCol->push_back("");
				break;
			default:
				state = 20;
				curCol += line[i];
				break;
			}
			break;
		case 20:
			switch (line[i])
			{
			case ',':
				state = 10;
				pRetCol->push_back(curCol);
				curCol = "";
				break;
			case '\r':
			case '\n':
				state = 50;
				pRetCol->push_back(curCol);
				curCol = "";
				break;
			default:
				curCol += line[i];
				break;
			}
			break;
		case 30:
			switch (line[i])
			{
			case '"':
				state = 40;
				break;
			case '\r':
			case '\n':
				state = 100;
				break;
			default:
				curCol += line[i];
				break;
			}
			break;
		case 40:
			switch (line[i])
			{
			case ',':
				state = 10;
				pRetCol->push_back(curCol);
				curCol = "";
				break;
			case '"':
				state = 30;
				curCol +="\"";
				break;
			case '\r':
			case '\n':
				state = 50;
				pRetCol->push_back(curCol);
				curCol = "";
				break;
			default:
				state = 100;
				break;
			}
			break;
		}
		if (state >= 50)
		{
			if (state==100)
				pRetCol->clear();
			break;
		}
	}
	return pRetCol;
}