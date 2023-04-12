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

from simData import *


# path = "../simulations/mnistBaseline/" 
# name = "MNIST"

path = "../simulations/cifarBaseline/" 
name = "CIFAR"

data = SimData(figureFolderPath="%s/figures/" % path, title="Mysims")

data.createSimulationStructureFromPattern( \
    path \
    # "../simulations/mnistBaseline/" \
    , name \
    # , "MNIST" \
    ,[] \
    , range(0,5)) 


# data.createSimulationStructureFromPattern( \
    # "../simulations/cifarBaseline/" \
    # , "CIFAR" \
    # ,[] \
    # , range(0,5)) 

Utils.ConfigUtil.loadConfigsForSimulations(data)

dataGroups = [ 
    ["task1TrainData", "task1ValidData"]
    # ["task1TrainData", "task1TrainDataBlur2", "task1TrainDataBlur10", "task1TrainDataBlur20", "task1TrainDataBlur40"]
    # , ["task1ValidData", "task1ValidDataBlur2", "task1ValidDataBlur10", "task1ValidDataBlur20", "task1ValidDataBlur40"]
    # , ["task1ValidDataSaltNoise1", "task1ValidDataSaltNoise5", "task1ValidDataSaltNoise10"]
    # , ["task1ValidDataSaltGaussianNoise1", "task1ValidDataSaltGaussianNoise10", "task1ValidDataSaltGaussianNoise20"]
]

for datasetNames in dataGroups:

    # for total dataset
    # classNames = [str(x) for x in range(20)] # includes class specific metrics
    classNames = []
    classNames.extend(["macroavg", "weightedavg"])
    metricSuffixs = ["precision", "recall", "f1-score"]
    metricNames = []
    metricFiles = []
    for className in classNames:
        for metricSuffix in metricSuffixs:
            metricNames.append("%s-%s" % (className, metricSuffix))
            metricFiles.append("/classReport/%s-%s.txt" % (className, metricSuffix))

    metricNames = ["matlabAcc", "loss", "ece", "mce"]
    metricFiles = ["matlabAccuracy.txt", "loss.txt", "ece.txt", "mce.txt"]

    for metricName, metricFile in zip(metricNames, metricFiles):
        Metrics.LoadData.loadMetric(data, metricName=metricName, metricFile=metricFile, forceDatasetLoadFolders=datasetNames, detectMemberDataFolders=False)
        # Metrics.Basics.plotTrialMetrics(data, datsetNames=datasetNames, metricNames=[metricName])
        Metrics.Basics.plotAvgSimMetrics([data], datsetNames=datasetNames, metricNames=[metricName])
        # Metrics.Basics.barTrialMetrics(data, datsetNames=datasetNames, metricNames=[metricName], plotIdxs=[0,-1], xticks=["Start Training", "End Training"])

    Confidence.loadMetric(data, datasetFolders=datasetNames)
    Confidence.plotConfidenceHistograms(data, datasetFolders=datasetNames, plotIdxs=[0,-1])


data.saveFigures()