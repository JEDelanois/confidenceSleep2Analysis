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

# figureFolderPath = "../icmlFigures/"
figureFolderPath = "../icmlFigures/"


datas = []

timePointsPrettyNames = []
lineColors = []
timePoints = []
lineStyles = []

baseline = SimData(figureFolderPath=figureFolderPath)
baseline.createSimulationStructureFromPattern( \
    "/bazhlab/edelanois/cnnSleep/89/simulationSweep-Baseline/" \
    , "Mnist Baseline" \
    ,[] \
    , range(0,10)) 
datas.append(baseline)
timePointsPrettyNames.append("Baseline")
lineColors.append("tab:blue")
timePoints.append(-1)
lineStyles.append("-")

baselineSleep = SimData(figureFolderPath=figureFolderPath)
baselineSleep.createSimulationStructureFromPattern( \
    "/bazhlab/edelanois/cnnSleep/89/simulationSweep-BaselineSleep/" \
    , "Mnist baselineSleep" \
    ,[] \
    , range(0,10)) 
datas.append(baselineSleep)
# timePointsPrettyNames.append("Baseline + SRC")
timePointsPrettyNames.append("SRC")
lineColors.append("tab:gray")
timePoints.append(-1)
lineStyles.append("-")

# baselineGradExp = SimData(figureFolderPath=figureFolderPath)
# baselineGradExp.createSimulationStructureFromPattern( \
    # "/bazhlab/edelanois/cnnSleep/89/simulationSweep-BaselineGradExpansion" \
    # , "Mnist baselineGradExp" \
    # ,[] \
    # , range(0,3)) 
# datas.append(baselineGradExp)
# # timePointsPrettyNames.append("Baseline + Grad Exp")
# timePointsPrettyNames.append("Grad Exp")
# lineColors.append("tab:green")
# timePoints.append(-1)
# lineStyles.append("-")

baselineSleepFf = SimData(figureFolderPath=figureFolderPath)
baselineSleepFf.createSimulationStructureFromPattern( \
    "/bazhlab/edelanois/cnnSleep/89/simulationSweep-BaselineSleepFf/" \
    , "Mnist baselineSleepFf" \
    ,[] \
    , range(0,10)) 
datas.append(baselineSleepFf)
# timePointsPrettyNames.append("Baseline + SRC + FFF")
timePointsPrettyNames.append("SRC + FFF")
lineColors.append("tab:orange")
timePoints.append(-1)
lineStyles.append("-")

# baselineFtSp = SimData(figureFolderPath=figureFolderPath)
# baselineFtSp.createSimulationStructureFromPattern( \
    # "/bazhlab/edelanois/cnnSleep/89/simulationSweep-BaselineFinetune/Sp/dataPercentage-1.0/" \
    # , "Mnist baselineFtSp" \
    # ,[] \
    # , range(0,10)) 
# datas.append(baselineFtSp)
# # timePointsPrettyNames.append("Baseline + Ft Sp")
# timePointsPrettyNames.append("Ft Sp")
# lineColors.append("tab:purple")
# timePoints.append(-1)
# lineStyles.append("--")

# baselineFtBlur = SimData(figureFolderPath=figureFolderPath)
# baselineFtBlur.createSimulationStructureFromPattern( \
    # "/bazhlab/edelanois/cnnSleep/89/simulationSweep-BaselineFinetune/Blur/dataPercentage-1.0/" \
    # , "Mnist baselineFtBlur" \
    # ,[] \
    # , range(0,10)) 
# datas.append(baselineFtBlur)
# # timePointsPrettyNames.append("Baseline + Ft Blur")
# timePointsPrettyNames.append("Ft Blur")
# lineColors.append("tab:brown")
# timePoints.append(-1)
# lineStyles.append("--")

# finetuneBlur = SimData(figureFolderPath=figureFolderPath)
# finetuneBlur.createSimulationStructureFromPattern( \
    # "/bazhlab/edelanois/cnnSleep/81/simulationSweep/Blur/dataPercentage-1.0/" \
    # , "Mnist finetuneBlur" \
    # ,[] \
    # , range(0,3)) 
# datas.append(finetuneBlur)
# # timePointsPrettyNames.append("Baseline + FT Blur")
# timePointsPrettyNames.append("FT Blur")
# lineColors.append("tab:brown")
# timePoints.append(-1)
# lineStyles.append("--")

# finetuneGn = SimData(figureFolderPath=figureFolderPath)
# finetuneGn.createSimulationStructureFromPattern( \
    # "/bazhlab/edelanois/cnnSleep/81/simulationSweep/Sp/dataPercentage-1.0/" \
    # , "Mnist finetuneGn" \
    # ,[] \
    # , range(0,3)) 
# datas.append(finetuneGn)
# # timePointsPrettyNames.append("Baseline + FT Sp")
# timePointsPrettyNames.append("FT Sp")
# lineColors.append("tab:purple")
# timePoints.append(-1)
# lineStyles.append("--")

for data in datas:
    Utils.ConfigUtil.loadConfigsForSimulations(data)

dataGroups = []
valueGroups = []

# Validation sets
# dataGroups.append(["task1ValidData", "task1ValidData-Blur-1", "task1ValidData-Blur-2", "task1ValidData-Blur-3", "task1ValidData-Blur-4", "task1ValidData-Blur-5", "task1ValidData-Blur-6",])
# valueGroups.append([0., 1., 2., 3., 4., 5., 6.])

# dataGroups.append(["task1ValidData", "task1ValidData-SP-0_1", "task1ValidData-SP-0_2", "task1ValidData-SP-0_3", "task1ValidData-SP-0_4", "task1ValidData-SP-0_5", "task1ValidData-SP-0_6",])
# valueGroups.append([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

# dataGroups.append(["task1ValidData", "task1ValidData-GN-0_1", "task1ValidData-GN-0_2", "task1ValidData-GN-0_3", "task1ValidData-GN-0_4", "task1ValidData-GN-0_5", "task1ValidData-GN-0_6",])
# valueGroups.append([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])


# Test sets
dataGroups.append(["testTask1AllData", "testTask1AllData-Blur-1", "testTask1AllData-Blur-2", "testTask1AllData-Blur-3", "testTask1AllData-Blur-4", "testTask1AllData-Blur-5", "testTask1AllData-Blur-6",])
valueGroups.append([0., 1., 2., 3., 4., 5., 6.])

dataGroups.append(["testTask1AllData", "testTask1AllData-SP-0_1", "testTask1AllData-SP-0_2", "testTask1AllData-SP-0_3", "testTask1AllData-SP-0_4", "testTask1AllData-SP-0_5", "testTask1AllData-SP-0_6",])
valueGroups.append([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

dataGroups.append(["testTask1AllData", "testTask1AllData-GN-0_1", "testTask1AllData-GN-0_2", "testTask1AllData-GN-0_3", "testTask1AllData-GN-0_4", "testTask1AllData-GN-0_5", "testTask1AllData-GN-0_6",])
valueGroups.append([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])




# # Extra sets
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
    # metricNames = ["loss", "matlabAcc"]
    # metricFiles = ["loss.txt", "matlabAccuracy.txt"]
    metricNames = ["matlabAcc"]
    metricFiles = ["matlabAccuracy.txt"]
    print(datasetNames)
    for metricName, metricFile in zip(metricNames, metricFiles):
        for data in datas:
            Metrics.LoadData.loadMetric(data, metricName=metricName, metricFile=metricFile, forceDatasetLoadFolders=datasetNames, detectMemberDataFolders=False)
        Metrics.Basics.plotSpecificTrialMetricOverDatasetValue(datas, datsetNames=datasetNames, datsetValues=valueGroup, timePoints=timePoints, metricName=metricName, timePointsPrettyNames=timePointsPrettyNames ,usePrettyXTicks=True, prettyFileName=None, prettyXLabel=None, lineStyles=lineStyles, lineColors=lineColors, alpha=0.2)
        Metrics.Basics.plotBarSpecificTrialMetricOverDatasetValue(datas, datsetNames=datasetNames, datsetValues=valueGroup, timePoints=timePoints, metricName=metricName, timePointsPrettyNames=timePointsPrettyNames ,usePrettyXTicks=True, prettyFileName=None, prettyXLabel=None, lineStyles=lineStyles, lineColors=lineColors, alpha=0.2)

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
