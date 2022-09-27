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

forceReload = True

simDatas = []

baseline = SimData(figureFolderPath="../figures/" )
baseline.createSimulationStructureFromPattern( \
    "/bazhlab/edelanois/cnnSleep/60/simulations/baseline/0/" \
    , "Baseline" \
    ,[] \
    , range(0,1)) 
simDatas.append(baseline)

fineTuneAll = SimData(figureFolderPath="../figures/" )
fineTuneAll.createSimulationStructureFromPattern( \
    "../simulations/genAlgo-All/sim808/" \
    , "Fine Tune All" \
    ,[] \
    , range(0,1)) 
simDatas.append(fineTuneAll)

fineTuneBlur = SimData(figureFolderPath="../figures/" )
fineTuneBlur.createSimulationStructureFromPattern( \
    "../simulations/genAlgo-Blur/sim524/" \
    , "Fine Tune Blur" \
    ,[] \
    , range(0,1)) 
simDatas.append(fineTuneBlur)

fineTuneGn = SimData(figureFolderPath="../figures/" )
fineTuneGn.createSimulationStructureFromPattern( \
    "../simulations/genAlgo-Gn/sim382/" \
    , "Fine Tune Gn" \
    ,[] \
    , range(0,1)) 
simDatas.append(fineTuneGn)

fineTuneSp = SimData(figureFolderPath="../figures/" )
fineTuneSp.createSimulationStructureFromPattern( \
    "../simulations/genAlgo-Sp/sim658/" \
    , "Fine Tune Sp" \
    ,[] \
    , range(0,1)) 
simDatas.append(fineTuneSp)

for s in simDatas:
    Utils.ConfigUtil.loadConfigsForSimulations(s)
# Utils.ConfigUtil.loadConfigsForSimulations(cnnSleepData)
# Utils.ConfigUtil.loadConfigsForSimulations(ffSleepData)

dataGroups = []
valueGroups = []

def getGroupName(dg):
    return dg[1].split("-")[1]

dataGroups.append(["task1ValidData", "task1ValidData-Blur-1", "task1ValidData-Blur-2", "task1ValidData-Blur-3", "task1ValidData-Blur-4", "task1ValidData-Blur-5", "task1ValidData-Blur-6",])
valueGroups.append([0., 1., 2., 3., 4., 5., 6.])

dataGroups.append(["task1ValidData", "task1ValidData-SP-0_1", "task1ValidData-SP-0_2", "task1ValidData-SP-0_3", "task1ValidData-SP-0_4", "task1ValidData-SP-0_5", "task1ValidData-SP-0_6",])
valueGroups.append([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

dataGroups.append(["task1ValidData", "task1ValidData-GN-0_1", "task1ValidData-GN-0_2", "task1ValidData-GN-0_3", "task1ValidData-GN-0_4", "task1ValidData-GN-0_5", "task1ValidData-GN-0_6",])
valueGroups.append([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

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
allDatasets = []
for dd in dataGroups: # dont add duplicates
    for d in dd:
        if d not in allDatasets:
            allDatasets.append(d)

for i,datasetNames in enumerate(dataGroups):
    valueGroup = valueGroups[i]
    metricNames = ["loss", "matlabAcc"]
    metricFiles = ["loss.txt", "matlabAccuracy.txt"]
    for metricName, metricFile in zip(metricNames, metricFiles):
        for s in simDatas:
            Metrics.LoadData.loadMetric(s, metricName=metricName, metricFile=metricFile, forceDatasetLoadFolders=datasetNames, detectMemberDataFolders=False)
        # Metrics.LoadData.loadMetric(cnnSleepData, metricName=metricName, metricFile=metricFile, forceDatasetLoadFolders=datasetNames, detectMemberDataFolders=False)
        # Metrics.LoadData.loadMetric(ffSleepData, metricName=metricName, metricFile=metricFile, forceDatasetLoadFolders=datasetNames, detectMemberDataFolders=False)

        # Metrics.Basics.plotTrialMetrics(fineTuneAll, datsetNames=datasetNames, metricNames=[metricName])
        # Metrics.Basics.plotTrialMetrics(cnnSleepData, datsetNames=datasetNames, metricNames=[metricName])
        # Metrics.Basics.plotTrialMetrics(ffSleepData, datsetNames=datasetNames, metricNames=[metricName])

        # Metrics.Basics.plotTrialMetricOverDatasetValue(fineTuneAll, datsetNames=datasetNames, datsetValues=valueGroup, timePoints=[0,-1], metricName=metricName, timePointsPrettyNames=["Post Training", "Post Finetuning"], prettyFileName=None)
        # Metrics.Basics.plotTrialMetricOverDatasetValue(cnnSleepData, datsetNames=datasetNames, datsetValues=valueGroup, timePoints=[0,-1], metricName=metricName, timePointsPrettyNames=["Post Training", "Post Finetuning"], prettyFileName=None)
        # Metrics.Basics.plotTrialMetricOverDatasetValue(ffSleepData, datsetNames=datasetNames, datsetValues=valueGroup, timePoints=[0,-1], metricName=metricName, timePointsPrettyNames=["Post Training", "Post Finetuning"], prettyFileName=None)


# datas = [fineTuneAll, fineTuneAll, cnnSleepData, ffSleepData]
# timePoints = [0, 1, 1, 1]
# timePointsPrettyNames = ["Post Training", "Sleep", "Sleep CNN", "Sleep FF"]

datas = [baseline, fineTuneAll, fineTuneAll, fineTuneBlur, fineTuneBlur, fineTuneGn, fineTuneGn, fineTuneSp, fineTuneSp]
timePoints = [0, 0, -1, 0, -1, 0, -1, 0, -1]
timePointsPrettyNames = ["Baseline", "FineTune All", "FineTune All + Sleep", "FineTune Blur", "FineTune Blur + Sleep", "FineTune Gn", "FineTune Gn + Sleep", "FineTune Sp", "FineTune Sp + Sleep"]
lineStyles = ["-", "--", "-", "--", "-", "--", "-", "--", "-"]
lineColors = ["k", "tab:blue", "tab:blue", "tab:orange", "tab:orange", "tab:green", "tab:green", "tab:red", "tab:red"]

for i,datasetNames1 in enumerate(dataGroups):
    groupName1 = getGroupName(datasetNames1)
    Metrics.Basics.plotMetricTable(datas, datsetNames=datasetNames1, timePoints=timePoints, metricName="matlabAcc", timePointsPrettyNames=timePointsPrettyNames, prettyXTicks=True, prettyFileName=groupName1, prettyXLabel=None)
    Metrics.Basics.plotSpecificTrialMetricOverDatasetValue(
        datas
        , datsetNames=datasetNames1
        , datsetValues=valueGroup
        , timePoints=timePoints 
        , metricName="matlabAcc"
        , timePointsPrettyNames=timePointsPrettyNames 
        , prettyXTicks=True
        , prettyFileName="%s-lineMatlabAcc.png" % groupName1
        , prettyXLabel=None
        ,lineStyles=lineStyles
        ,lineColors=lineColors)
    # Metrics.Basics.plotMetricTable(datas, datsetNames=datasetNames1, timePoints=timePoints, metricName="microF1", timePointsPrettyNames=timePointsPrettyNames, prettyXTicks=True, prettyFileName=groupName1, prettyXLabel=None)
    # Metrics.Basics.plotMetricTable(datas, datsetNames=datasetNames1, timePoints=timePoints, metricName="weightedF1", timePointsPrettyNames=timePointsPrettyNames, prettyXTicks=True, prettyFileName=groupName1, prettyXLabel=None)

Metrics.Basics.plotMetricTable(datas, datsetNames=allDatasets, timePoints=timePoints, metricName="matlabAcc", timePointsPrettyNames=timePointsPrettyNames, prettyXTicks=True, prettyFileName="allTable", prettyXLabel=None)

fineTuneAll.saveFigures()
# ffSleepData.saveFigures()
# fineTuneAll.saveFigures()
# cnnSleepData.saveFigures()
