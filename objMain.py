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

from simData import *


data = SimData(figureFolderPath="../figures/" )


data.createSimulationStructureFromPattern( \
    "../simulations/MseLoss5/" \
    , "Sequential Training" \
    ,[] \
    , range(0,1)) 

# data.createSimulationStructureSweepFolder( \
    # "../simulationSweep/OLDHiddenClassesSleep/" \
    # , "Hidden Sleep" \
    # , titlePatternSameAsFilePattern=False)

print(len(data.sims))
Utils.ConfigUtil.loadConfigsForSimulations(data)

# Metrics.LoadData.loadConfidences(data)
# Metrics.LoadData.loadClassMetric2(data)

# metricNames = ["TP", "TN", "FP", "FN", "precision", "recall", "accuracy", "loss", "matlabAccuracy"] 
# metricFiles = ["TFPN/TP.txt", "TFPN/TN.txt", "TFPN/FP.txt", "TFPN/FN.txt", "TFPN/precision.txt", "TFPN/recall.txt", "accuracy.txt", "loss.txt", "matlabAccuracy.txt"] 
# datasetNames = ["Training", "task1TrainData", "task2TrainData"]

# for metricName, metricFile in zip(metricNames, metricFiles):
    # Metrics.LoadData.loadMetric(data, metricName=metricName, metricFile=metricFile, forceDatasetLoadFolders=datasetNames, detectMemberDataFolders=False)
    # for datasetName in datasetNames:
        # Metrics.Basics.plotMetric(data, datsetName=datasetName, metricName=metricName)

# metricNames = ["TP", "TN", "FP", "FN", "precision", "recall", "F1", "accuracy", "loss", "matlabAccuracy"] 
# metricFiles = ["TFPN/TP.txt", "TFPN/TN.txt", "TFPN/FP.txt", "TFPN/FN.txt", "TFPN/precision.txt", "TFPN/recall.txt", "TFPN/F1Score.txt", "accuracy.txt", "loss.txt", "matlabAccuracy.txt"] 
# datasetNames = ["task1TrainData", "task2TrainData"]

metricNames = ["TP", "TN", "FP", "FN", "precision", "recall", "F1", "accuracy", "loss", "matlabAccuracy"] 
metricFiles = ["TFPN/TP.txt", "TFPN/TN.txt", "TFPN/FP.txt", "TFPN/FN.txt", "TFPN/precision.txt", "TFPN/recall.txt", "TFPN/F1Score.txt", "accuracy.txt", "loss.txt", "matlabAccuracy.txt"] 
datasetNames = ["task1TrainData", "task1TrainDataBlur2", "task1TrainDataBlur10", "task1TrainDataBlur20", "task1TrainDataBlur40"]

for metricName, metricFile in zip(metricNames, metricFiles):
    Metrics.LoadData.loadMetric(data, metricName=metricName, metricFile=metricFile, forceDatasetLoadFolders=datasetNames, detectMemberDataFolders=False)
    Metrics.Basics.plotTrialMetrics(data, datsetNames=datasetNames, metricNames=[metricName])
    Metrics.Basics.barTrialMetrics(data, datsetNames=datasetNames, metricNames=[metricName], plotIdxs=[0,-1], xticks=["Post Training", "Post Sleep"])

# for total dataset
classNames = [str(x) for x in range(4)]
classNames.extend(["microavg", "macroavg", "weightedavg"])
metricSuffixs = ["precision", "recall", "f1-score"]
metricNames = []
metricFiles = []
for className in classNames:
    for metricSuffix in metricSuffixs:
        metricNames.append("%s-%s" % (className, metricSuffix))
        metricFiles.append("%s-%s.txt" % (className, metricSuffix))

# load all class report
datasetNames = ["classReport"]
for metricName, metricFile in zip(metricNames, metricFiles):
    Metrics.LoadData.loadMetric(data, metricName=metricName, metricFile=metricFile, forceDatasetLoadFolders=datasetNames, detectMemberDataFolders=False)

# load individual class reports
tempDatasetNames = ["task1TrainData", "task1TrainDataBlur2", "task1TrainDataBlur10", "task1TrainDataBlur20", "task1TrainDataBlur40"]
metricFiles = []
for className in classNames:
    for metricSuffix in metricSuffixs:
        metricFiles.append("/classReport/%s-%s.txt" % (className, metricSuffix))

for metricName, metricFile in zip(metricNames, metricFiles):
    Metrics.LoadData.loadMetric(data, metricName=metricName, metricFile=metricFile, forceDatasetLoadFolders=tempDatasetNames, detectMemberDataFolders=False)

datasetNames.extend(tempDatasetNames)

for metricName in metricNames:
    Metrics.Basics.plotTrialMetrics(data, datsetNames=datasetNames, metricNames=[metricName])
    Metrics.Basics.barTrialMetrics(data, datsetNames=datasetNames, metricNames=[metricName], plotIdxs=[0,-1], xticks=["Start Training", "End Training"])

data.saveFigures()
