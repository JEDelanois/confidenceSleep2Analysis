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
import Feature.FeatureMain

from simData import *


datas = []

baseline = SimData(figureFolderPath="../figures/" )
baseline.createSimulationStructureFromPattern( \
    "../simulationSweep-Baseline/" \
    , "Mnist Baseline" \
    ,[] \
    , range(0,3)) 
datas.append(baseline)

baselineSleep = SimData(figureFolderPath="../figures/" )
baselineSleep.createSimulationStructureFromPattern( \
    "../simulationSweep-BaselineSleep/" \
    , "Mnist baselineSleep" \
    ,[] \
    , range(0,3)) 
datas.append(baselineSleep)

baselineGradExp = SimData(figureFolderPath="../figures/" )
baselineGradExp.createSimulationStructureFromPattern( \
    "../simulationSweep-BaselineGradExpansion/" \
    , "Mnist baselineGradExp" \
    ,[] \
    , range(0,3)) 
datas.append(baselineGradExp)

for data in datas:
    Utils.ConfigUtil.loadConfigsForSimulations(data)

dataGroups = []
valueGroups = []

dataGroups.append(["testTask1AllData", "testTask1AllData-Blur-1", "testTask1AllData-Blur-2", "testTask1AllData-Blur-3", "testTask1AllData-Blur-4", "testTask1AllData-Blur-5", "testTask1AllData-Blur-6",])
valueGroups.append([0., 1., 2., 3., 4., 5., 6.])

dataGroups.append(["testTask1AllData", "testTask1AllData-SP-0_1", "testTask1AllData-SP-0_2", "testTask1AllData-SP-0_3", "testTask1AllData-SP-0_4", "testTask1AllData-SP-0_5", "testTask1AllData-SP-0_6",])
valueGroups.append([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

dataGroups.append(["testTask1AllData", "testTask1AllData-GN-0_1", "testTask1AllData-GN-0_2", "testTask1AllData-GN-0_3", "testTask1AllData-GN-0_4", "testTask1AllData-GN-0_5", "testTask1AllData-GN-0_6",])
valueGroups.append([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

# dataGroups.append(["testTask1AllData", "testTask1AllData-SE-0_25", "testTask1AllData-SE-0_50", "testTask1AllData-SE-0_60", "testTask1AllData-SE-0_75", "testTask1AllData-SE-0_90", "testTask1AllData-SE-1_0",])
# valueGroups.append([0.0, 0.25, 0.50, 0.60, 0.75, 0.90, 1.0])

# dataGroups.append(["testTask1AllData", "testTask1AllData-PN",])
# valueGroups.append([0.0, 1.1])

# dataGroups.append(["testTask1AllData", "testTask1AllData-Dark-0_0", "testTask1AllData-Dark-0_16", "testTask1AllData-Dark-0_33", "testTask1AllData-Dark-0_5", "testTask1AllData-Dark-0_66", "testTask1AllData-Dark-0_83",]) # dark-0_0 is just black and screws up plots
# valueGroups.append([])

#  dataGroups.append(["testTask1AllData", "testTask1AllData-Dark-0_16", "testTask1AllData-Dark-0_33", "testTask1AllData-Dark-0_5", "testTask1AllData-Dark-0_66", "testTask1AllData-Dark-0_83",])
#  valueGroups.append([0.0, 0.16, 0.33, 0.5, 0.66, 0.83])

#  dataGroups.append(["testTask1AllData", "testTask1AllData-Bright-1_5", "testTask1AllData-Bright-2_0", "testTask1AllData-Bright-2_5", "testTask1AllData-Bright-3_0", "testTask1AllData-Bright-3_5", "testTask1AllData-Bright-4_0"])
#  valueGroups.append([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])


# dataGroups.append(["testTask1AllData", "testTask1AllData-Blur-1", "testTask1AllData-Blur-2", "testTask1AllData-Blur-3"])
# dataGroups.append(["testTask1AllData", "testTask1AllData-SP-0_1", "testTask1AllData-SP-0_2", "testTask1AllData-SP-0_3"])
# dataGroups.append(["testTask1AllData", "testTask1AllData-GN-0_1", "testTask1AllData-GN-0_2", "testTask1AllData-GN-0_3"])
# dataGroups.append(["testTask1AllData", "testTask1AllData-SE-0_25", "testTask1AllData-SE-0_50", "testTask1AllData-SE-0_75"])
# dataGroups.append(["testTask1AllData", "testTask1AllData-PN"])

# dataGroups.append(["task1TrainData", "task1TrainDataBlur2", "task1TrainDataBlur10", "task1TrainDataBlur20", "task1TrainDataBlur40"]) 
# dataGroups.append(["task1TrainData", "task1TrainDataBlur2", "testTask1AllData"]) 
# dataGroups.append(["testTask1AllData", "task1ValidDataBlur1", "task1ValidDataBlur1_5", "task1ValidDataBlur2", "task1ValidDataBlur2_5", "task1ValidDataBlur3"])
# dataGroups.append(["testTask1AllData", "task1ValidDataSP0-1", "task1ValidDataSP0-25", "task1ValidDataSP0-5"])


# # for trainDataset in  [True, False]:

# for i,datasetNames in enumerate(dataGroups):
    # valueGroup = valueGroups[i]
    # metricNames = ["loss", "matlabAcc"]
    # metricFiles = ["loss.txt", "matlabAccuracy.txt"]
    # print(datasetNames)
    # for metricName, metricFile in zip(metricNames, metricFiles):
        # Metrics.LoadData.loadMetric(data, metricName=metricName, metricFile=metricFile, forceDatasetLoadFolders=datasetNames, detectMemberDataFolders=False)
        # Metrics.Basics.plotTrialMetrics(data, datsetNames=datasetNames, metricNames=[metricName])
        # Metrics.Basics.plotTrialMetricOverDatasetValue(data, datsetNames=datasetNames, datsetValues=valueGroup, timePoints=[-1], metricName=metricName, timePointsPrettyNames=["EndStage"], prettyFileName=None)

for i,datasetNames in enumerate(dataGroups):
    valueGroup = valueGroups[i]
    metricNames = ["loss", "matlabAcc"]
    metricFiles = ["loss.txt", "matlabAccuracy.txt"]
    print(datasetNames)
    for metricName, metricFile in zip(metricNames, metricFiles):
        for data in datas:
            Metrics.LoadData.loadMetric(data, metricName=metricName, metricFile=metricFile, forceDatasetLoadFolders=datasetNames, detectMemberDataFolders=False)
        Metrics.Basics.plotSpecificTrialMetricOverDatasetValue(datas, datsetNames=datasetNames, datsetValues=valueGroup, timePoints=[-1,-1, -1], metricName=metricName, timePointsPrettyNames=["Baseline", "Baseline + Sleep", "Baseline + Grad Exp"],usePrettyXTicks=True, prettyFileName=None, prettyXLabel=None, lineStyles=["-", "-", "-"], lineColors=["tab:blue", "tab:orange", "tab:green"], alpha=0.2)


# SleepAnalysis.SleepBasics.loadSleepData(data)
# SleepAnalysis.SleepBasics.plotSleepStuff(data)

# for i,dataGroup in enumerate(dataGroups):
    # Feature.FeatureMain.getConvLayerGradientMetric(
        # data
        # , modelName="model"
        # , modelNameloadModelPaths=["/stateDict/modelStateDict0.pth", "/stateDict/modelStateDict1.pth"]
        # , linePrettyNames = ["Post Training", "Post Sleep"]
        # , datasetNames=dataGroup
        # , datsetValues=valueGroups[i]
        # )
    # break

datas[0].saveFigures()
