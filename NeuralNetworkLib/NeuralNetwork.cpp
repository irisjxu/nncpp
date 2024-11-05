#include "pch.h"
#include "NeuralNetworkLib.h"

using namespace Eigen;

void NeuralNetwork::CreateDatasetFromCSV(string folderPath, const vector<ColIOType>& vecColIoType, const vector<bool>& vecIsCatCol, DataSetAlloc& dsAlloc, bool ignoreFirstRow /* = true*/, const vector<string>& colNames  /*=vector<string>() */)
{
	pDataset = make_unique<Dataset>();

	pDataset->OpenCsvDataFile(folderPath, vecIsCatCol, vecColIoType, dsAlloc, ignoreFirstRow, colNames);
	pDataset->CompleteLoadDataSrc();
	pDataset->UniformStandardize();

	numOutputLayerNodes = pDataset->GetOutputFeatureNum();

	numInputLayerNodes = pDataset->GetInputFeatureNum();
}

void NeuralNetwork::AddHiddenLayer(int numNodes, ActivationFunction& activationFunction, InitializationFunction& weightInitFunc, InitializationFunction& biasInitFunc)
{
	int numPrevNodes;
	if (hiddenLayers.empty())
		numPrevNodes = numInputLayerNodes;
	else
		numPrevNodes = hiddenLayers.back().numNodes;
	hiddenLayers.push_back(Layer(numNodes, numPrevNodes, &activationFunction, &weightInitFunc, &biasInitFunc));
}

void NeuralNetwork::AddOutputLayer(ActivationFunction& activationFunction, InitializationFunction& weightInitFunc, InitializationFunction& biasInitFunc)
{
	int numPrevNodes;
	if (hiddenLayers.empty())
		numPrevNodes = numInputLayerNodes;
	else
		numPrevNodes = hiddenLayers.back().numNodes;
	pOutputLayer = make_unique<OutputLayer>(numOutputLayerNodes, numPrevNodes, &activationFunction, &weightInitFunc, &biasInitFunc);
}

void NeuralNetwork::SetParameters(int numEpochs, float learningRate, int batchSize, LossFunction::LossFunctionTypes lossFunctionType)
{
	this->numEpochs = numEpochs;
	this->learningRate = learningRate;
	this->batchSize = batchSize;
	lossFunction.lossFuncType = lossFunctionType;
}

bool NeuralNetwork::BeginTraining()
{
	thread trainThread([this] {this->TrainInThread(); });
	trainThread.detach();
	std::this_thread::sleep_for(std::chrono::milliseconds(1000));
	return true;
}

void NeuralNetwork::TrainInThread()
{
	modelStatusMutex.lock();
	modelStat.running = true;
	modelStatusMutex.unlock();


	for (int epoch = 0; epoch < numEpochs; epoch++)
	{
		pDataset->ResetTrainingEpoch();
		float trainingLoss = 0;
		int numTrain = 0;
		float* pInputDataRow;
		float* pOutputDataRow;

		while (pDataset->RetrieveNextTrainingRcd(&pInputDataRow, &pOutputDataRow))
		{
			numTrain++;
			Map<VectorXf> input(pInputDataRow, numInputLayerNodes);
			Map<VectorXf> target(pOutputDataRow, numOutputLayerNodes);
			VectorXf output(numOutputLayerNodes);

			//forward:
			Forward(input, output, true);
			trainingLoss += lossFunction.CalculateLossScoreForward(output, target);

			//backward:
			VectorXf gradient = lossFunction.CalculateLossScoreBackward(output, target);
			pOutputLayer->Backward(gradient, gradient);
			MatrixXf* prevWeights = &(pOutputLayer->weights);
			for (std::list<Layer>::reverse_iterator rit = hiddenLayers.rbegin(); rit != hiddenLayers.rend(); rit++)
			{
				rit->Backward(gradient, prevWeights, gradient);
				prevWeights = &(rit->weights);
			}
			if (numTrain % batchSize == 0)
				updateNetwork(batchSize);
		}
		if (numTrain % batchSize != 0)
			updateNetwork(numTrain % batchSize);

		//validation:
		float numCorrect = 0;
		float numVal = 0;
		float valLoss = 0;
		pDataset->ResetValidationRcds();

		while (pDataset->RetrieveNextValidRcd(&pInputDataRow, &pOutputDataRow))
		{
			Map<VectorXf> input(pInputDataRow, numInputLayerNodes);
			Map<VectorXf> target(pOutputDataRow, numOutputLayerNodes);
			VectorXf output(numOutputLayerNodes);
			Forward(input, output, false);

			valLoss += lossFunction.CalculateLossScoreForward(output, target);

			//accuracy for single categorical outome
			//Eigen::Index predictedIdx;
			//float max = output.maxCoeff(&predictedIdx);
			//if (target(predictedIdx) == float(1))
			//	numCorrect++;

			// accuracy for binary outcome:
			//if (output(0) > 0.5 && target(0) == 1 || output(0) <= 0.5 && target(0) == 0)
			//	numCorrect++;
			numVal++;
		}

		//adjust learning rate
		if (epoch % 10 == 0)
			learningRate *= (float)0.97;

		modelStatusMutex.lock();
		ModelResult rst;
		rst.epochs = epoch + 1;
		rst.trainLoss = trainingLoss / numTrain;
		rst.validLoss = valLoss / numVal;
		//rst.validAccuracy = numCorrect / numTotal;
		rst.validAccuracy = rst.validLoss;
		modelStat.listRst.push_back(rst);
		modelStatusMutex.unlock();
	}

	//test
	float* pInputDataRow;
	unique_ptr<vector<vector<float>>> testOutputs = make_unique<vector<vector<float>>>();
	while (pDataset->RetrieveNextTestRcd(&pInputDataRow))
	{
		Map<VectorXf> input(pInputDataRow, numInputLayerNodes);
		VectorXf output(numOutputLayerNodes);
		Forward(input, output, false);
		testOutputs->push_back(vector<float>(output.data(), output.data() + output.size()));
	}

	pDataset->UniformDestandardize(*testOutputs);

	modelStatusMutex.lock();
	modelStat.pTestPredictVals = move(testOutputs);
	modelStat.running = false;
	modelStatusMutex.unlock();
}

void NeuralNetwork::Forward(Map<VectorXf>& input, VectorXf& output, bool alterNetwork)
{
	if (alterNetwork)
	{
		hiddenLayers.front().Forward(input, output);
		for (auto it = ++hiddenLayers.begin(); it != hiddenLayers.end(); it++)
			it->Forward(output, output);
		pOutputLayer->Forward(output, output);
	}
	else
	{
		hiddenLayers.front().ForwardCalculate(input, output);
		for (auto it = ++hiddenLayers.begin(); it != hiddenLayers.end(); it++)
			it->ForwardCalculate(output, output);
		pOutputLayer->ForwardCalculate(output, output);
	}
}

void NeuralNetwork::updateNetwork(int numSample)
{
	for (auto it = hiddenLayers.begin(); it != hiddenLayers.end(); it++)
	{
		it->weights -= it->weightGradientSum / numSample * learningRate;
		it->biases -= it->biasGradientSum / numSample * learningRate;
		//cout << it->weightGradientSum / batchSize << "\n\n";
		//it->weights = (it->weights - it->weightGradientSum / batchSize * learningRate);
		//it->biases = (it->biases - it->biasGradientSum / batchSize * learningRate);

		it->weightGradientSum.setZero();
		it->biasGradientSum.setZero();

	}

	pOutputLayer->weights -= pOutputLayer->weightGradientSum / numSample * learningRate;
	pOutputLayer->biases -= pOutputLayer->biasGradientSum / numSample * learningRate;
	//cout << pOutputLayer->weightGradientSum / batchSize << "\n\n";

	pOutputLayer->weightGradientSum.setZero();
	pOutputLayer->biasGradientSum.setZero();

}

void NeuralNetwork::RetrieveResult(ModelStatus& tr)
{
	modelStatusMutex.lock();
	tr.running = modelStat.running;
	tr.pTestPredictVals = make_unique<vector<vector<float>>>();
	*(tr.pTestPredictVals) = *(modelStat.pTestPredictVals);
	tr.listRst.splice(tr.listRst.end(), modelStat.listRst);
	modelStatusMutex.unlock();
}

