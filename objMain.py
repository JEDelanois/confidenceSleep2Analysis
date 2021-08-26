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
    "/bazhlab/edelanois/objectDetection/projects/objectDetection/8/simulations/sim1" \
    , "Sim 1 test" \
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

metricNames = ["TP", "TN", "FP", "FN", "precision", "recall", "accuracy", "loss"] 
metricFiles = ["TFPN/TP.txt", "TFPN/TN.txt", "TFPN/FP.txt", "TFPN/FN.txt", "TFPN/precision.txt", "TFPN/recall.txt", "accuracy.txt", "loss.txt"] 
datasetNames = ["Training"]

for metricName, metricFile in zip(metricNames, metricFiles):
    Metrics.LoadData.loadMetric(data, metricName=metricName, metricFile=metricFile, forceDatasetLoadFolders=datasetNames)
    Metrics.Basics.plotMetric(data, datsetName=datasetNames[0], metricName=metricName)

data.saveFigures()