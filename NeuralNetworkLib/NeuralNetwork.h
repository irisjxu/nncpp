#pragma once

class NeuralNetwork
{
public:
	NeuralNetwork() {}

	void CreateDatasetFromCSV(string folderPath, const vector<ColIOType>& vecColIoType, const vector<bool>& vecIsCatCol, DataSetAlloc& dsAlloc, bool ignoreFirstRow = true, const vector<string>& colNames = vector<string>());
	//void addCSVDatafile(string folderPath, DataSetAlloc& dsAlloc, bool ignoreFirstRow = true) {}

	void AddHiddenLayer(int numNodes, ActivationFunction& activationFunction, InitializationFunction& weightInitFunc, InitializationFunction& biasInitFunc);
	void AddOutputLayer(ActivationFunction& activationFunction, InitializationFunction& weightInitFunc, InitializationFunction& biasInitFunc);

	void SetParameters(int numEpochs, float learningRate, int batchSize, LossFunction::LossFunctionTypes lossFunctionType);

	bool BeginTraining();

	void RetrieveResult(ModelStatus& tr);
	void RetrieveTestTarget(vector<vector<float>>& vecOutputData) { pDataset->RetrieveTestTarget(vecOutputData); }

protected:
	mutex modelStatusMutex;
	ModelStatus modelStat;

	void TrainInThread();
	void updateNetwork(int numSample);
	void Forward(Map<VectorXf>& input, VectorXf& output, bool alterNetwork);

	int numEpochs = 100;
	float learningRate = (float)0.01;
	int batchSize = 32;
	LossFunction lossFunction;

	unique_ptr<Dataset> pDataset;

	int numInputLayerNodes = 0;
	int numOutputLayerNodes = 0;
	list<Layer> hiddenLayers;
	unique_ptr<OutputLayer> pOutputLayer;
};
