#pragma once
#include "pch.h"


#include <Eigen/Dense>
#include <algorithm>
#include <thread>
#include <future>
#include <memory>
#include <string>
#include <iostream>
#include <cstring>
#include <vector>
#include <filesystem>
#include <random> 

enum class ColIOType;
class DataSetAlloc;
class DataColumn;

#include "..\\CsvLib\\CsvLib.h"
#include "DataSrc.h"
#include "Dataset.h"
#include "DataSetAlloc.h"

#include "ActivationFunction.h"
#include "InitializationFunction.h"
#include "Layer.h"
#include "LossFunction.h"
#include "ModelStatus.h"
#include "NeuralNetwork.h"
