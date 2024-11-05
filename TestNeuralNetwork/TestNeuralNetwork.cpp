/*
* Flow of using the NeuralNetworkLib:
*
* - construct a model with NeuralNetwork constructor
*
* - create a dataset from file using CreateDatasetFromCSV method in NeuralNetwork
* - (to be implemented) add data to the dataset from file *optional*
*
* - add hidden layers one at a time using AddHiddenLayer method in NeuralNetwork
* - add output layer using AddOutputLayer in NeuralNetwork class
*
* - set model hyperparameters using SetParameters method in NeuralNetwork
*
* - being model training
*
* - check current status of model anytime using RetrieveResult method in NeuralNetwork
* - retreive test dataset target values using RetrieveTestTarget method in NeuralNetwork
*/


/*
* Info about dataset creation
*
* Use the CreateDatasetFromCSV from your NeuralNetwork object
* Parameters:
* - FILE PATH: string
* - I/O COLUMN TYPES: vector<colIoType> - each element is either ColIOType::Input or ColIOType::Output, corresponding to each column in the csv file
* - CAT/NUM COLUMN TYPES: vector<bool> - each element is true if the column in the csv file is categorical, and false if numerical
* - SPLIT DATASET: DataSetAlloc - object that specifies how to split the dataset across train/val/test (DataSetAllocRatio is the only currently implemented way)
* - IGNORE FIRST ROW: bool (default = true) - specifies if the first row isn't a data sample
* - COLUMN NAMES: vector<string> (default from first row of file) - each element is the name of the corresponding column in the csv file
*/


//TWO SAMPLE USES OF LIBRARY:

#include "../NeuralNetworkLib/NeuralNetworkLib.h"
using namespace std;
using namespace Eigen;
using namespace chrono;


//Blueberry dataset example
void Blueberry()
{
	srand(3250);
	NeuralNetwork model;

	vector<bool> catColumns(17, false);
	vector<ColIOType> colIoTypes(17, ColIOType::Input);
	colIoTypes[16] = ColIOType::Output;
	DataSetAllocRatio splitDataset(8, 1, 1);

	model.CreateDatasetFromCSV("blueberryNoId.csv"
		, colIoTypes
		, catColumns
		, splitDataset);

	ReluActivationFunction relu;
	LinearActivationFunction linearAf;
	NormalDistInitializationFunction heInit;
	NormalDistInitializationFunction xavierInit(1);
	ConstantInitializationFunction biasInit;
	model.AddHiddenLayer(15, relu, xavierInit, biasInit);
	model.AddHiddenLayer(10, relu, xavierInit, biasInit);
	model.AddOutputLayer(linearAf, xavierInit, biasInit);

	model.SetParameters(75, (float)0.1, 32, LossFunction::LossFunctionTypes::MSE);

	model.BeginTraining();

	ModelStatus tr;
	while (true)
	{
		auto a = *tr.pTestPredictVals;
		model.RetrieveResult(tr);
		if (!tr.running)
			break;
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		for (auto& rst : tr.listRst)
			cout << rst.epochs << ", " << rst.trainLoss << ", " << rst.validLoss << "\n";

		tr.listRst.clear();
	}

	vector<vector<float>>testTarget;
	model.RetrieveTestTarget(testTarget);

	model.RetrieveResult(tr);
	//tr.pTestPredictVals
	cout << "Test result: target, predicted" << endl;
	for (int i = 0; i < testTarget.size(); i++)
		cout << testTarget[i][0] << ", " << (*tr.pTestPredictVals)[i][0] << endl;
}

//breast cancer dataset example
void BreastCancer()
{
	NeuralNetwork model;

	vector<bool> catColumns(31, false);
	catColumns[30] = true;
	vector<ColIOType> colIoTypes(31, ColIOType::Input);
	colIoTypes[30] = ColIOType::Output;
	//vector<string> colNames = { "SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm","Species" };
	DataSetAllocRatio splitDataset(8, 2, 0);

	model.CreateDatasetFromCSV("breast-cancer.csv"
		, colIoTypes
		, catColumns
		, splitDataset);

	ReluActivationFunction relu;
	SigmoidWithCrossEntropyActivationFunction sigmoid;
	NormalDistInitializationFunction heInit;
	NormalDistInitializationFunction xavierInit(1);
	ConstantInitializationFunction biasInit;
	model.AddHiddenLayer(16, relu, heInit, biasInit);
	model.AddHiddenLayer(16, relu, heInit, biasInit);
	model.AddOutputLayer(sigmoid, xavierInit, biasInit);

	model.SetParameters(500, (float)0.08, 32, LossFunction::LossFunctionTypes::CrossEntropyWithSigmoid);

	model.BeginTraining();

	ModelStatus tr;
	while (true)
	{
		model.RetrieveResult(tr);
		if (!tr.running)
			break;
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		for (auto& rst : tr.listRst)
			cout << rst.epochs << ", " << rst.trainLoss << ", " << rst.validLoss <<", "<< rst.validAccuracy << "\n";
		tr.listRst.clear();
	}
}

int main()
{
	Blueberry();
	//BreastCancer();
}