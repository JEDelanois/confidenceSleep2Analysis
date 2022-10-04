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
import code

def getGroupName(dg):
    if len(dg) == 1:
        return dg[0]
    return dg[1].split("-")[1]

datas = []
datasNames = []
lineData = []
xValuess = []

dataAll = SimData(figureFolderPath="../figures/" )
dataAll.createSimulationStructureSweepFolder(
    "/bazhlab/edelanois/cnnSleep/68/simulationSweep/All/", 
    "", 
    titlePatternSameAsFilePattern=False
    )
datasNames.append("Finetune All Distortions")
lineData.append({})
lineData[-1]["style"] = "--"
lineData[-1]["color"] = "tab:blue"
datas.append(dataAll)
xValuess.append(datas[-1].paramPathValueSetDicts[1]["values"])

dataBlur = SimData(figureFolderPath="../figures/" )
dataBlur.createSimulationStructureSweepFolder(
    "/bazhlab/edelanois/cnnSleep/68/simulationSweep/Blur/", 
    "", 
    titlePatternSameAsFilePattern=False
    )
datasNames.append("Finetune Blur")
lineData.append({})
lineData[-1]["style"] = "--"
lineData[-1]["color"] = "tab:orange"
datas.append(dataBlur)
xValuess.append(datas[-1].paramPathValueSetDicts[1]["values"])

dataGn = SimData(figureFolderPath="../figures/" )
dataGn.createSimulationStructureSweepFolder(
    "/bazhlab/edelanois/cnnSleep/68/simulationSweep/Gn/", 
    "", 
    titlePatternSameAsFilePattern=False
    )
datasNames.append("Finetune Gn")
lineData.append({})
lineData[-1]["style"] = "--"
lineData[-1]["color"] = "tab:green"
datas.append(dataGn)
xValuess.append(datas[-1].paramPathValueSetDicts[1]["values"])

dataSp = SimData(figureFolderPath="../figures/" )
dataSp.createSimulationStructureSweepFolder(
    "/bazhlab/edelanois/cnnSleep/68/simulationSweep/Sp/", 
    "", 
    titlePatternSameAsFilePattern=False
    )
datasNames.append("Finetune Sp")
lineData.append({})
lineData[-1]["style"] = "--"
lineData[-1]["color"] = "tab:red"
datas.append(dataSp)
xValuess.append(datas[-1].paramPathValueSetDicts[1]["values"])
# # ____________________________________

# dataAllSleep = SimData(figureFolderPath="../figures/" )
# dataAllSleep.createSimulationStructureSweepFolder(
    # "../simulationSweep/All/", 
    # "", 
    # titlePatternSameAsFilePattern=False
    # )
# datasNames.append("Finetune All Distortions + Sleep")
# lineData.append({})
# lineData[-1]["style"] = "-"
# lineData[-1]["color"] = "tab:blue"
# datas.append(dataAllSleep)
# xValuess.append(datas[-1].paramPathValueSetDicts[1]["prettyValues"])

# dataBlurSleep = SimData(figureFolderPath="../figures/" )
# dataBlurSleep.createSimulationStructureSweepFolder(
    # "../simulationSweep/Blur/", 
    # "", 
    # titlePatternSameAsFilePattern=False
    # )
# datasNames.append("Finetune Blur + Sleep")
# lineData.append({})
# lineData[-1]["style"] = "-"
# lineData[-1]["color"] = "tab:orange"
# datas.append(dataBlurSleep)
# xValuess.append(datas[-1].paramPathValueSetDicts[1]["prettyValues"])

# dataGnSleep = SimData(figureFolderPath="../figures/" )
# dataGnSleep.createSimulationStructureSweepFolder(
    # "../simulationSweep/Gn/", 
    # "", 
    # titlePatternSameAsFilePattern=False
    # )
# datasNames.append("Finetune Gn + Sleep")
# lineData.append({})
# lineData[-1]["style"] = "-"
# lineData[-1]["color"] = "tab:green"
# datas.append(dataGnSleep)
# xValuess.append(datas[-1].paramPathValueSetDicts[1]["prettyValues"])

# dataSpSleep = SimData(figureFolderPath="../figures/" )
# dataSpSleep.createSimulationStructureSweepFolder(
    # "../simulationSweep/Sp/", 
    # "", 
    # titlePatternSameAsFilePattern=False
    # )
# datasNames.append("Finetune Sp + Sleep")
# lineData.append({})
# lineData[-1]["style"] = "-"
# lineData[-1]["color"] = "tab:red"
# datas.append(dataSpSleep)
# xValuess.append(datas[-1].paramPathValueSetDicts[1]["prettyValues"])


for data in datas:
    Utils.ConfigUtil.loadConfigsForSimulations(data)


dataGroups = []
valueGroups = []

dataGroups.append(["task1ValidData"])
valueGroups.append([0.])

dataGroups.append(["task1ValidData", "task1ValidData-Blur-1", "task1ValidData-Blur-2", "task1ValidData-Blur-3", "task1ValidData-Blur-4", "task1ValidData-Blur-5", "task1ValidData-Blur-6",])
valueGroups.append([0., 1., 2., 3., 4., 5., 6.])

dataGroups.append(["task1ValidData", "task1ValidData-Blur-1", "task1ValidData-Blur-2", "task1ValidData-Blur-3", "task1ValidData-Blur-4", "task1ValidData-Blur-5", "task1ValidData-Blur-6",])
valueGroups.append([0., 1., 2., 3., 4., 5., 6.])

dataGroups.append(["task1ValidData", "task1ValidData-SP-0_1", "task1ValidData-SP-0_2", "task1ValidData-SP-0_3", "task1ValidData-SP-0_4", "task1ValidData-SP-0_5", "task1ValidData-SP-0_6",])
valueGroups.append([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

dataGroups.append(["task1ValidData", "task1ValidData-GN-0_1", "task1ValidData-GN-0_2", "task1ValidData-GN-0_3", "task1ValidData-GN-0_4", "task1ValidData-GN-0_5", "task1ValidData-GN-0_6",])
valueGroups.append([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

#  dataGroups.append(["task1ValidData", "task1ValidData-SE-0_25", "task1ValidData-SE-0_50", "task1ValidData-SE-0_60", "task1ValidData-SE-0_75", "task1ValidData-SE-0_90", "task1ValidData-SE-1_0",])
#  valueGroups.append([0.0, 0.25, 0.50, 0.60, 0.75, 0.90, 1.0])

#  dataGroups.append(["task1ValidData", "task1ValidData-PN",])
#  valueGroups.append([0.0, 1.1])

# dataGroups.append(["task1ValidData", "task1ValidData-Dark-0_0", "task1ValidData-Dark-0_16", "task1ValidData-Dark-0_33", "task1ValidData-Dark-0_5", "task1ValidData-Dark-0_66", "task1ValidData-Dark-0_83",]) # dark-0_0 is just black and screws up plots
# valueGroups.append([])

#  dataGroups.append(["task1ValidData", "task1ValidData-Dark-0_16", "task1ValidData-Dark-0_33", "task1ValidData-Dark-0_5", "task1ValidData-Dark-0_66", "task1ValidData-Dark-0_83",])
#  valueGroups.append([0.0, 0.16, 0.33, 0.5, 0.66, 0.83])

#  dataGroups.append(["task1ValidData", "task1ValidData-Bright-1_5", "task1ValidData-Bright-2_0", "task1ValidData-Bright-2_5", "task1ValidData-Bright-3_0", "task1ValidData-Bright-3_5", "task1ValidData-Bright-4_0"])
#  valueGroups.append([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])

allDatasets = []
for dd in dataGroups: # dont add duplicates
    for d in dd:
        if d not in allDatasets:
            allDatasets.append(d)


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
        for data in datas:
            Metrics.LoadData.loadMetric(data, metricName=metricName, metricFile=metricFile, forceDatasetLoadFolders=datasetNames, detectMemberDataFolders=False)
            # Metrics.Basics.plotTrialMetrics(data, datsetNames=datasetNames, metricNames=[metricName])
            # Metrics.Basics.plotTrialMetricOverDatasetValue(data, datsetNames=datasetNames, datsetValues=valueGroup, timePoints=[0,-1], metricName=metricName, timePointsPrettyNames=["Post Training", "Post Sleep"], prettyFileName=None)

for i,datasetNames in enumerate(dataGroups):
    gn = getGroupName(datasetNames)
    func = Metrics.Basics.meanPerformanceAtTimeGenerator(timePoint=-1, datasetNames=datasetNames, metricName="matlabAcc", alpha=0.1)
    Metrics.Basics.plotMetricOverConfigValue(datas, xValuess=xValuess, configPath=["modifiers", 1, 1,"datasetPercentages", 0], simPerformanceFunction=func, prettyFileName="performance_%s.png" % gn, datasNames=datasNames, lineData=lineData, ylabel="matlabAcc", xlabel="Dataset Size", xscale="log", title="%s Perofrmance" % gn)

func = Metrics.Basics.meanPerformanceAtTimeGenerator(timePoint=-1, datasetNames=allDatasets, metricName="matlabAcc", alpha=0.1)
Metrics.Basics.plotMetricOverConfigValue(datas, xValuess=xValuess, configPath=["modifiers", 1, 1,"datasetPercentages", 0], simPerformanceFunction=func, prettyFileName="performance_%s.png" % "All", datasNames=datasNames, lineData=lineData, ylabel="matlabAcc", xlabel="Dataset Size", xscale="log", title="%s Perofrmance" % "All")

func = Metrics.Basics.meanPerformanceAtTimeGenerator(timePoint=-1, datasetNames=allDatasets, metricName="matlabAcc", alpha=0.1)
Metrics.Basics.plotMetricOverConfigValue(datas, xValuess=xValuess, configPath=["modifiers", 1, 1,"datasetPercentages", 0], simPerformanceFunction=func, prettyFileName="performance_%s.png" % "All", datasNames=datasNames, lineData=lineData, ylabel="matlabAcc", xlabel="Dataset Size", xscale="log", title="%s Perofrmance" % "All")

for data in datas:
    data.saveFigures()
