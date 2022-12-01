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

def createMultiMetricGroup(datasetForm="testTask1AllData-Blur-%s-SP-%s", dataset0Vals=[1, 3, 6], dataset1Vals=[0.1, 0.3, 0.6]):

    def prettyNum(num):
        return str(num).replace(".", "_")

    ret = []
    for d0 in dataset0Vals:
        for d1 in dataset1Vals:
            ret.append(datasetForm % (prettyNum(d0), prettyNum(d1)))
    return ret


data = SimData(figureFolderPath="../figures/" )

data.createSimulationStructureFromPattern( \
    "../simulationSweep-Baseline/" \
    , "Baseline" \
    ,[] \
    , range(0,3)) 

data.createSimulationStructureFromPattern( \
    "../simulationSweep-BaselineGradExpansion/" \
    , "Grad Expansion" \
    ,[] \
    , range(0,3)) 

data.createSimulationStructureFromPattern( \
    "../simulationSweep-BaselineSleep/" \
    , "SRC" \
    ,[] \
    , range(0,3)) 

data.createSimulationStructureFromPattern( \
    "../simulationSweep-BaselineFinetuneBlur/" \
    , "Baseline + FT Blur" \
    ,[] \
    , range(0,3)) 

data.createSimulationStructureFromPattern( \
    "../simulationSweep-BaselineFinetuneGN/" \
    , "Baseline + FT GN" \
    ,[] \
    , range(0,3)) 

data.createSimulationStructureFromPattern( \
    "../simulationSweep-BaselineFinetuneSP/" \
    , "Baseline + FT SP" \
    ,[] \
    , range(0,3)) 

Utils.ConfigUtil.loadConfigsForSimulations(data)

dataGroups = []
valueGroups = []

dataGroups.append(["testTask1AllData", "testTask1AllData-Blur-1", "testTask1AllData-Blur-2", "testTask1AllData-Blur-3", "testTask1AllData-Blur-4", "testTask1AllData-Blur-5", "testTask1AllData-Blur-6",])
valueGroups.append([0., 1., 2., 3., 4., 5., 6.])

dataGroups.append(["testTask1AllData", "testTask1AllData-SP-0_1", "testTask1AllData-SP-0_2", "testTask1AllData-SP-0_3", "testTask1AllData-SP-0_4", "testTask1AllData-SP-0_5", "testTask1AllData-SP-0_6",])
valueGroups.append([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

dataGroups.append(["testTask1AllData", "testTask1AllData-GN-0_1", "testTask1AllData-GN-0_2", "testTask1AllData-GN-0_3", "testTask1AllData-GN-0_4", "testTask1AllData-GN-0_5", "testTask1AllData-GN-0_6",])
valueGroups.append([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])


# dataGroups.append(["task1ValidData", "task1ValidData-Blur-1", "task1ValidData-Blur-2", "task1ValidData-Blur-3", "task1ValidData-Blur-4", "task1ValidData-Blur-5", "task1ValidData-Blur-6",])
# valueGroups.append([0., 1., 2., 3., 4., 5., 6.])

#  dataGroups.append(["task1ValidData", "task1ValidData-SP-0_1", "task1ValidData-SP-0_2", "task1ValidData-SP-0_3", "task1ValidData-SP-0_4", "task1ValidData-SP-0_5", "task1ValidData-SP-0_6",])
#  valueGroups.append([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

#  dataGroups.append(["task1ValidData", "task1ValidData-GN-0_1", "task1ValidData-GN-0_2", "task1ValidData-GN-0_3", "task1ValidData-GN-0_4", "task1ValidData-GN-0_5", "task1ValidData-GN-0_6",])
#  valueGroups.append([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

# dataGroups.append(["task1ValidData", "task1ValidData-SE-0_25", "task1ValidData-SE-0_50", "task1ValidData-SE-0_60", "task1ValidData-SE-0_75", "task1ValidData-SE-0_90", "task1ValidData-SE-1_0",])
# valueGroups.append([0.0, 0.25, 0.50, 0.60, 0.75, 0.90, 1.0])

# dataGroups.append(["task1ValidData", "task1ValidData-PN",])
# valueGroups.append([0.0, 1.1])

# dataGroups.append(["task1ValidData", "task1ValidData-Dark-0_0", "task1ValidData-Dark-0_16", "task1ValidData-Dark-0_33", "task1ValidData-Dark-0_5", "task1ValidData-Dark-0_66", "task1ValidData-Dark-0_83",]) # dark-0_0 is just black and screws up plots
# valueGroups.append([])

# dataGroups.append(["task1ValidData", "task1ValidData-Dark-0_16", "task1ValidData-Dark-0_33", "task1ValidData-Dark-0_5", "task1ValidData-Dark-0_66", "task1ValidData-Dark-0_83",])
# valueGroups.append([0.0, 0.16, 0.33, 0.5, 0.66, 0.83])

# dataGroups.append(["task1ValidData", "task1ValidData-Bright-1_5", "task1ValidData-Bright-2_0", "task1ValidData-Bright-2_5", "task1ValidData-Bright-3_0", "task1ValidData-Bright-3_5", "task1ValidData-Bright-4_0"])
# valueGroups.append([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])


# dataGroups.append(["task1ValidData", "task1ValidData-Blur-1", "task1ValidData-Blur-2", "task1ValidData-Blur-3"])
# dataGroups.append(["task1ValidData", "task1ValidData-SP-0_1", "task1ValidData-SP-0_2", "task1ValidData-SP-0_3"])
# dataGroups.append(["task1ValidData", "task1ValidData-GN-0_1", "task1ValidData-GN-0_2", "task1ValidData-GN-0_3"])
# dataGroups.append(["task1ValidData", "task1ValidData-SE-0_25", "task1ValidData-SE-0_50", "task1ValidData-SE-0_75"])
# dataGroups.append(["task1ValidData", "task1ValidData-PN"])

# dataGroups.append(["task1TrainData", "task1TrainDataBlur2", "task1TrainDataBlur10", "task1TrainDataBlur20", "task1TrainDataBlur40"]) 
# dataGroups.append(["task1TrainData", "task1TrainDataBlur2", "task1ValidData"]) 
# dataGroups.append(["task1ValidData", "task1ValidDataBlur1", "task1ValidDataBlur1_5", "task1ValidDataBlur2", "task1ValidDataBlur2_5", "task1ValidDataBlur3"])
# dataGroups.append(["task1ValidData", "task1ValidDataSP0-1", "task1ValidDataSP0-25", "task1ValidDataSP0-5"])


# # for trainDataset in  [True, False]:

for i,datasetNames in enumerate(dataGroups):
    valueGroup = valueGroups[i]
    metricNames = ["loss", "matlabAcc"]
    metricFiles = ["loss.txt", "matlabAccuracy.txt"]
    print(datasetNames)
    for metricName, metricFile in zip(metricNames, metricFiles):
        Metrics.LoadData.loadMetric(data, metricName=metricName, metricFile=metricFile, forceDatasetLoadFolders=datasetNames, detectMemberDataFolders=False)
        # Metrics.Basics.plotTrialMetrics(data, datsetNames=datasetNames, metricNames=[metricName])
        # Metrics.Basics.plotTrialMetricOverDatasetValue(data, datsetNames=datasetNames, datsetValues=valueGroup, timePoints=[-1], metricName=metricName, timePointsPrettyNames=["See Sim Folder"], prettyFileName=None)

multiMetricGroups = []
multiMetricGroups.append(createMultiMetricGroup(datasetForm="testTask1AllData-Blur-%s-SP-%s", dataset0Vals=[1, 3, 6], dataset1Vals=[0.1, 0.3, 0.6]))
multiMetricGroups.append(createMultiMetricGroup(datasetForm="testTask1AllData-SP-%s-Blur-%s", dataset1Vals=[1, 3, 6], dataset0Vals=[0.1, 0.3, 0.6]))
multiMetricGroups.append(["testTask1AllData"])

for datasetNames in multiMetricGroups:
    metricNames = ["loss", "matlabAcc"]
    metricFiles = ["loss.txt", "matlabAccuracy.txt"]
    print(datasetNames)
    for metricName, metricFile in zip(metricNames, metricFiles):
        Metrics.LoadData.loadMetric(data, metricName=metricName, metricFile=metricFile, forceDatasetLoadFolders=datasetNames, detectMemberDataFolders=False)

# Metrics.Basics.plotMultiMetricTable(data, datasetForm="testTask1AllData-Blur-%s-SP-%s", dataset0Vals=[1, 3, 6], dataset1Vals=[0.1, 0.3, 0.6], timePoint=0, metricName="matlabAcc", timePointsPrettyName=None, prettyFileName=None)
Metrics.Basics.plotMultiMetricTable(
    data
    , datasetForm="testTask1AllData-Blur-%s-SP-%s"

    , dataset0Vals=[0, 1, 3, 6]
    , dataset1Vals=[0, 0.1, 0.3, 0.6]

    , dataset0_0Fallback="testTask1AllData-SP-%s"
    , dataset1_0Fallback="testTask1AllData-Blur-%s"

    , both_0Fallback="testTask1AllData"

    , data0Label="Blur Intensity"
    , data1Label="SP Intensity"

    , timePoint=0
    , metricName="matlabAcc"
    , timePointsPrettyName=None
    , prettyFileName=None

    ,vmin=0.0
    ,vmax=1.0
    )

Metrics.Basics.plotMultiMetricTable(
    data
    , datasetForm="testTask1AllData-SP-%s-Blur-%s"

    , dataset1Vals=[0, 1, 3, 6]
    , dataset0Vals=[0, 0.1, 0.3, 0.6]

    , dataset1_0Fallback="testTask1AllData-SP-%s"
    , dataset0_0Fallback="testTask1AllData-Blur-%s"

    , both_0Fallback="testTask1AllData"

    , data0Label="SP Intensity"
    , data1Label="Blur Intensity"

    , timePoint=0
    , metricName="matlabAcc"
    , timePointsPrettyName=None
    , prettyFileName=None

    ,vmin=0.0
    ,vmax=1.0
    )

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

data.saveFigures()
