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
import code

from simData import *


fullSleepData = SimData(figureFolderPath="../figures/" )
fullSleepData.createSimulationStructureFromPattern( \
    "/bazhlab/edelanois/cnnSleep/6/simulations/multiDistortionTest/" \
    , "Cnn Sleep" \
    ,[] \
    , range(0,1)) 

cnnSleepData = SimData(figureFolderPath="../figures/" )
cnnSleepData.createSimulationStructureFromPattern( \
    "/bazhlab/edelanois/cnnSleep/6/simulations/multiDistortionTest-cnnPlasticity/" \
    , "Cnn Sleep" \
    ,[] \
    , range(0,1)) 

ffSleepData = SimData(figureFolderPath="../figures/" )
ffSleepData.createSimulationStructureFromPattern( \
    "/bazhlab/edelanois/cnnSleep/6/simulations/multiDistortionTest-ffPlasticity/" \
    , "Cnn Sleep" \
    ,[] \
    , range(0,1)) 



Utils.ConfigUtil.loadConfigsForSimulations(fullSleepData)
Utils.ConfigUtil.loadConfigsForSimulations(cnnSleepData)
Utils.ConfigUtil.loadConfigsForSimulations(ffSleepData)

dataGroups = []
valueGroups = []

def getGroupName(dg):
    return dg[1].split("-")[1]

dataGroups.append(["task1ValidData", "task1ValidData-Blur-1", "task1ValidData-Blur-2", "task1ValidData-Blur-3", "task1ValidData-Blur-4", "task1ValidData-Blur-5", "task1ValidData-Blur-6",])
valueGroups.append([0., 1., 2., 3., 4., 5., 6.])

dataGroups.append(["task1ValidData", "task1ValidData-SP-0_1", "task1ValidData-SP-0_2", "task1ValidData-SP-0_3", "task1ValidData-SP-0_4", "task1ValidData-SP-0_5", "task1ValidData-SP-0_6",])
valueGroups.append([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5])

dataGroups.append(["task1ValidData", "task1ValidData-GN-0_1", "task1ValidData-GN-0_2", "task1ValidData-GN-0_3", "task1ValidData-GN-0_4", "task1ValidData-GN-0_5", "task1ValidData-GN-0_6",])
valueGroups.append([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5])

dataGroups.append(["task1ValidData", "task1ValidData-SE-0_25", "task1ValidData-SE-0_50", "task1ValidData-SE-0_60", "task1ValidData-SE-0_75", "task1ValidData-SE-0_90", "task1ValidData-SE-1_0",])
valueGroups.append([0.0, 0.25, 0.50, 0.60, 0.75, 0.90, 1.0])

dataGroups.append(["task1ValidData", "task1ValidData-PN",])
valueGroups.append([0.0, 1.1])

# dataGroups.append(["task1ValidData", "task1ValidData-Dark-0_0", "task1ValidData-Dark-0_16", "task1ValidData-Dark-0_33", "task1ValidData-Dark-0_5", "task1ValidData-Dark-0_66", "task1ValidData-Dark-0_83",]) # dark-0_0 is just black and screws up plots
# valueGroups.append([])

dataGroups.append(["task1ValidData", "task1ValidData-Dark-0_16", "task1ValidData-Dark-0_33", "task1ValidData-Dark-0_5", "task1ValidData-Dark-0_66", "task1ValidData-Dark-0_83",])
valueGroups.append([0.0, 0.16, 0.33, 0.5, 0.66, 0.83])

dataGroups.append(["task1ValidData", "task1ValidData-Bright-1_5", "task1ValidData-Bright-2_0", "task1ValidData-Bright-2_5", "task1ValidData-Bright-3_0", "task1ValidData-Bright-3_5", "task1ValidData-Bright-4_0"])
valueGroups.append([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])

allDatasets = []
[allDatasets.extend(d) for d in dataGroups]


picklePath = "./finetune.pkl"
quickloadedFinetune = False
if os.path.exists(picklePath):
    with open(picklePath, 'rb') as handle:
        allFinetune = pickle.load(handle)
        quickloadedFinetune = True
else:
    allFinetune = []
    for datasetname in allDatasets:
        if datasetname == "task1ValidData":
            continue
        curData = SimData(figureFolderPath="../figures/" )
        curData.createSimulationStructureFromPattern( \
            "/bazhlab/edelanois/cnnSleep/7/simulationSweep/finetuneSweep/dataset-%s/" % datasetname.replace("task1ValidData", "task1TrainData") \
            , datasetname \
            ,[] \
            , range(0,1)) 
        Utils.ConfigUtil.loadConfigsForSimulations(curData)
        allFinetune.append(curData)
        # code.interact(local=dict(globals(), **locals()))

for i,datasetNames in enumerate(dataGroups):
    valueGroup = valueGroups[i]
    metricNames = ["loss", "matlabAcc", "microF1", "weightedF1"]
    metricFiles = ["loss.txt", "matlabAccuracy.txt", "classReport/macroavg-f1-score.txt", "classReport/weightedavg-f1-score.txt"]
    for metricName, metricFile in zip(metricNames, metricFiles):
        Metrics.LoadData.loadMetric(fullSleepData, metricName=metricName, metricFile=metricFile, forceDatasetLoadFolders=datasetNames, detectMemberDataFolders=False)
        Metrics.LoadData.loadMetric(cnnSleepData, metricName=metricName, metricFile=metricFile, forceDatasetLoadFolders=datasetNames, detectMemberDataFolders=False)
        Metrics.LoadData.loadMetric(ffSleepData, metricName=metricName, metricFile=metricFile, forceDatasetLoadFolders=datasetNames, detectMemberDataFolders=False)
        if not quickloadedFinetune:
            for fine in allFinetune:
                Metrics.LoadData.loadMetric(fine, metricName=metricName, metricFile=metricFile, forceDatasetLoadFolders=datasetNames, detectMemberDataFolders=False)

        # Metrics.Basics.plotTrialMetrics(fullSleepData, datsetNames=datasetNames, metricNames=[metricName])
        # Metrics.Basics.plotTrialMetrics(cnnSleepData, datsetNames=datasetNames, metricNames=[metricName])
        # Metrics.Basics.plotTrialMetrics(ffSleepData, datsetNames=datasetNames, metricNames=[metricName])

        # Metrics.Basics.plotTrialMetricOverDatasetValue(fullSleepData, datsetNames=datasetNames, datsetValues=valueGroup, timePoints=[0,-1], metricName=metricName, timePointsPrettyNames=["Post Training", "Post Finetuning"], prettyFileName=None)
        # Metrics.Basics.plotTrialMetricOverDatasetValue(cnnSleepData, datsetNames=datasetNames, datsetValues=valueGroup, timePoints=[0,-1], metricName=metricName, timePointsPrettyNames=["Post Training", "Post Finetuning"], prettyFileName=None)
        # Metrics.Basics.plotTrialMetricOverDatasetValue(ffSleepData, datsetNames=datasetNames, datsetValues=valueGroup, timePoints=[0,-1], metricName=metricName, timePointsPrettyNames=["Post Training", "Post Finetuning"], prettyFileName=None)

if not quickloadedFinetune:
    with open(picklePath, 'wb') as handle:
        pickle.dump(allFinetune, handle, protocol=pickle.HIGHEST_PROTOCOL)

datas = [fullSleepData, fullSleepData, cnnSleepData, ffSleepData]
timePoints = [0, 1, 1, 1]
timePointsPrettyNames = ["Post Training", "Sleep", "Sleep CNN", "Sleep FF"]

for i,datasetNames in enumerate(dataGroups):
    groupName = getGroupName(datasetNames)
    extraDatas = [d for d in allFinetune if d.titlePattern in datasetNames]
    extraTimePoints = [-1] * len(extraDatas)
    extraTimePointsPrettyNames = [d.titlePattern.replace("task1TrainData", "task1ValidData") for d in extraDatas]
    Metrics.Basics.plotMetricTable(datas + extraDatas, datsetNames=datasetNames, timePoints=timePoints + extraTimePoints, metricName="matlabAcc", timePointsPrettyNames=timePointsPrettyNames + extraTimePointsPrettyNames, prettyXTicks=True, prettyFileName=groupName, prettyXLabel=None)
    Metrics.Basics.plotSpecificTrialMetricOverDatasetValue(
        datas + extraDatas
        , datsetNames=datasetNames
        , datsetValues=valueGroup
        , timePoints=timePoints + extraTimePoints
        , metricName="matlabAcc"
        , timePointsPrettyNames=timePointsPrettyNames + extraTimePointsPrettyNames
        , prettyXTicks=True
        , prettyFileName="%s-lineMatlabAcc.png" % groupName
        , prettyXLabel=None)
    # Metrics.Basics.plotMetricTable(datas, datsetNames=datasetNames, timePoints=timePoints, metricName="microF1", timePointsPrettyNames=timePointsPrettyNames, prettyXTicks=True, prettyFileName=groupName, prettyXLabel=None)
    # Metrics.Basics.plotMetricTable(datas, datsetNames=datasetNames, timePoints=timePoints, metricName="weightedF1", timePointsPrettyNames=timePointsPrettyNames, prettyXTicks=True, prettyFileName=groupName, prettyXLabel=None)

extraTimePoints = [-1] * len(allFinetune)
extraTimePointsPrettyNames = [d.titlePattern.replace("task1ValidData", "task1TrainData") for d in allFinetune]
Metrics.Basics.plotMetricTable(datas + allFinetune, datsetNames=allDatasets, timePoints=timePoints + extraTimePoints, metricName="matlabAcc", timePointsPrettyNames=timePointsPrettyNames + extraTimePointsPrettyNames, prettyXTicks=True, prettyFileName="allTable", prettyXLabel=None)

# Metrics.Basics.plotMetricTable(datas, datsetNames=allDatasets, timePoints=timePoints, metricName="microF1", timePointsPrettyNames=timePointsPrettyNames, prettyXTicks=True, prettyFileName="allTable", prettyXLabel=None)
# Metrics.Basics.plotMetricTable(datas, datsetNames=allDatasets, timePoints=timePoints, metricName="weightedF1", timePointsPrettyNames=timePointsPrettyNames, prettyXTicks=True, prettyFileName="allTable", prettyXLabel=None)


ffSleepData.saveFigures()
# fullSleepData.saveFigures()
# cnnSleepData.saveFigures()
