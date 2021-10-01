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

metricNames = ["TP", "TN", "FP", "FN", "precision", "recall", "F1", "accuracy", "loss", "matlabAccuracy"] 
metricFiles = ["TFPN/TP.txt", "TFPN/TN.txt", "TFPN/FP.txt", "TFPN/FN.txt", "TFPN/precision.txt", "TFPN/recall.txt", "TFPN/F1Score.txt", "accuracy.txt", "loss.txt", "matlabAccuracy.txt"] 
datasetNames = ["task1TrainData", "task2TrainData"]

for metricName, metricFile in zip(metricNames, metricFiles):
    Metrics.LoadData.loadMetric(data, metricName=metricName, metricFile=metricFile, forceDatasetLoadFolders=datasetNames, detectMemberDataFolders=False)
    Metrics.Basics.plotTrialMetrics(data, datsetNames=datasetNames, metricNames=[metricName])

data.saveFigures()