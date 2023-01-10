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

figureFolderPath = "../figures/"

allDatas = []
modelNameloadModelPaths=[]
modelPrettyNames = []

baselineData = SimData(figureFolderPath=figureFolderPath )
baselineData.createSimulationStructureFromPattern( \
    "/bazhlab/adahuja/code/95/simulationSweep-Baseline/" \
    , "Baseline" \
    ,[] \
    , range(0,1)) 
allDatas.append(baselineData)
modelNameloadModelPaths.append("/stateDict/modelStateDict50.pth")
modelPrettyNames.append("Baseline")

sleepData = SimData(figureFolderPath=figureFolderPath )
sleepData.createSimulationStructureFromPattern( \
    "/bazhlab/adahuja/code/95/simulationSweep-BaselineSleep/" \
    , "SRC" \
    ,[] \
    , range(0,1)) 
allDatas.append(sleepData)
modelNameloadModelPaths.append("/stateDict/modelStateDict1.pth")
modelPrettyNames.append("SRC")

sleepData = SimData(figureFolderPath=figureFolderPath )
sleepData.createSimulationStructureFromPattern( \
    "/bazhlab/edelanois/cnnSleep/81/simulationSweep/Sp/dataPercentage-1.0/" \
    , "Finetune SP" \
    ,[] \
    , range(0,10)) 
allDatas.append(sleepData)
modelNameloadModelPaths.append("/stateDict/modelStateDict10.pth")
modelPrettyNames.append("Finetune SP")

for d in allDatas:
    Utils.ConfigUtil.loadConfigsForSimulations(d)

dataGroups = []
valueGroups = []

dataGroups.append(["task1ValidData"])
valueGroups.append([0.])

# dataGroups.append(["task1ValidData", "task1ValidData-Blur-1", "task1ValidData-Blur-2", "task1ValidData-Blur-3", "task1ValidData-Blur-4", "task1ValidData-Blur-5", "task1ValidData-Blur-6"])
# valueGroups.append([0., 1., 2., 3., 4., 5., 6.])

# dataGroups.append(["task1ValidData", "task1ValidData-SP-0_1", "task1ValidData-SP-0_2", "task1ValidData-SP-0_3", "task1ValidData-SP-0_4", "task1ValidData-SP-0_5", "task1ValidData-SP-0_6"])
# valueGroups.append([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

# dataGroups.append(["task1ValidData", "task1ValidData-GN-0_1", "task1ValidData-GN-0_2", "task1ValidData-GN-0_3", "task1ValidData-GN-0_4", "task1ValidData-GN-0_5", "task1ValidData-GN-0_6"])
# valueGroups.append([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

# dataGroups.append(["task1ValidData", "task1ValidData-SE-0_25", "task1ValidData-SE-0_50", "task1ValidData-SE-0_60", "task1ValidData-SE-0_75", "task1ValidData-SE-0_90", "task1ValidData-SE-1_0"])
# valueGroups.append([0.0, 0.25, 0.50, 0.60, 0.75, 0.90, 1.0])

#  dataGroups.append(["task1ValidData", "task1ValidData-PN",])
#  valueGroups.append([0.0, 1.1])

# dataGroups.append(["task1ValidData", "task1ValidData-Dark-0_0", "task1ValidData-Dark-0_16", "task1ValidData-Dark-0_33", "task1ValidData-Dark-0_5", "task1ValidData-Dark-0_66", "task1ValidData-Dark-0_83"]) # dark-0_0 is just black and screws up plots
# valueGroups.append([])

#  dataGroups.append(["task1ValidData", "task1ValidData-Dark-0_16", "task1ValidData-Dark-0_33", "task1ValidData-Dark-0_5", "task1ValidData-Dark-0_66", "task1ValidData-Dark-0_83"])
#  valueGroups.append([0.0, 0.16, 0.33, 0.5, 0.66, 0.83])

#  dataGroups.append(["task1ValidData", "task1ValidData-Bright-1_5", "task1ValidData-Bright-2_0", "task1ValidData-Bright-2_5", "task1ValidData-Bright-3_0", "task1ValidData-Bright-3_5", "task1ValidData-Bright-4_0"])
#  valueGroups.append([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])

for dataGroup,valueGroup in zip(dataGroups, valueGroups):
    Feature.FeatureMain.getMultiGradcam(
    allDatas
    , modelName="model"
    , modelNameloadModelPaths=modelNameloadModelPaths
    , modelPrettyNames = modelPrettyNames
    , datasetNames=dataGroup
    , datsetValues=valueGroup
    # , imgIndexes=[0,1,2]
    , imgIndexes=[i for i in range(25)]
    )
    allDatas[0].saveFigures()
    allDatas[0].clearFigs()

# Feature.FeatureMain.getGradcam(
# allDatas
# , modelName="model"
# , modelNameloadModelPaths=["/stateDict/modelStateDict0.pth"]
# , linePrettyNames = ["Baseline"]
# , datasetNames=["task1ValidData"]
# , datsetValues=[0]
# , imgIndexes=[0,1,2,3,4,5,6]
# )
# data.saveFigures()
# data.clearFigs()

# for dataGroup,valueGroup in zip(dataGroups, valueGroups):
    # Feature.FeatureMain.getGradcam(
    # allDatas
    # , modelName="model"
    # , modelNameloadModelPaths=["/stateDict/modelStateDict50.pth", "/stateDict/modelStateDict1.pth"]
    # , modelPrettyNames = ["Baseline", "SRC"]
    # , datasetNames=dataGroup
    # , datsetValues=valueGroup
    # , imgIndexes=[0,1,2,3,4,5,6]
    # )
    # # Feature.FeatureMain.getGradcam(
    # # allDatas
    # # , modelName="model"
    # # , modelNameloadModelPaths=["/stateDict/modelStateDict0.pth"]
    # # , linePrettyNames = ["Post training"]
    # # , datasetNames=["task1ValidData", "task1ValidData-Blur-1", "task1ValidData-Blur-2", "task1ValidData-Blur-3"]
    # # , datsetValues=[0, 1, 2, 3]
    # # , imgIndexes=[0, 1]
    # # )
    # data.saveFigures()
    # data.clearFigs()


