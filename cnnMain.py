from os import wait
import matplotlib
matplotlib.use("agg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib._pylab_helpers


import Weights.LoadWeightData 
import Weights.Pca
import Weights.WeightBasics
import Utils.ConfigUtil
import Utils.RuntimeUtil
import Metrics.LoadData
import Metrics.Basics
import Confidence.Confidence as Confidence
import SleepAnalysis.SleepBasics

from simData import *


data = SimData(figureFolderPath="../figures/" )

data.createSimulationStructureFromPattern( \
    "/bazhlab/edelanois/cnnSleep/1/simulations/trainMnist1/" \
    # "/bazhlab/edelanois/cnnSleep/1/simulations/bestSleep/" \
    , "Cnn Sleep" \
    ,[] \
    , range(0,1)) 

Utils.ConfigUtil.loadConfigsForSimulations(data)

dataGroups = []
# dataGroups.append(["task1TrainData", "task1TrainDataBlur2", "task1TrainDataBlur10", "task1TrainDataBlur20", "task1TrainDataBlur40"]) 
dataGroups.append(["task1TrainData", "task1TrainDataBlur2", "task1ValidData"]) 
# dataGroups.append(["task1ValidData", "task1ValidDataBlur1", "task1ValidDataBlur1_5", "task1ValidDataBlur2", "task1ValidDataBlur2_5", "task1ValidDataBlur3"])
# dataGroups.append(["task1ValidData", "task1ValidDataSP0-1", "task1ValidDataSP0-25", "task1ValidDataSP0-5"])

# for trainDataset in  [True, False]:
for datasetNames in dataGroups:
    metricNames = ["loss", "matlabAcc"]
    metricFiles = ["loss.txt", "matlabAccuracy.txt"]
    print(datasetNames)
    for metricName, metricFile in zip(metricNames, metricFiles):
        Metrics.LoadData.loadMetric(data, metricName=metricName, metricFile=metricFile, forceDatasetLoadFolders=datasetNames, detectMemberDataFolders=False)
        Metrics.Basics.plotTrialMetrics(data, datsetNames=datasetNames, metricNames=[metricName])

data.saveFigures()
