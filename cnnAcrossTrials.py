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


fullSleepData = SimData(figureFolderPath="../figures/" )
fullSleepData.createSimulationStructureFromPattern( \
    "../simulations/multiDistortionTest/" \
    , "Cnn Sleep" \
    ,[] \
    , range(0,1)) 

cnnSleepData = SimData(figureFolderPath="../figures/" )
cnnSleepData.createSimulationStructureFromPattern( \
    "../simulations/multiDistortionTest-cnnPlasticity/" \
    , "Cnn Sleep" \
    ,[] \
    , range(0,1)) 

ffSleepData = SimData(figureFolderPath="../figures/" )
ffSleepData.createSimulationStructureFromPattern( \
    "../simulations/multiDistortionTest-ffPlasticity/" \
    , "Cnn Sleep" \
    ,[] \
    , range(0,1)) 

Utils.ConfigUtil.loadConfigsForSimulations(fullSleepData)
Utils.ConfigUtil.loadConfigsForSimulations(cnnSleepData)
Utils.ConfigUtil.loadConfigsForSimulations(ffSleepData)

dataGroups = []
valueGroups = []

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



for i,datasetNames in enumerate(dataGroups):
    valueGroup = valueGroups[i]
    metricNames = ["loss", "matlabAcc"]
    metricFiles = ["loss.txt", "matlabAccuracy.txt"]
    print(datasetNames)
    for metricName, metricFile in zip(metricNames, metricFiles):
        Metrics.LoadData.loadMetric(fullSleepData, metricName=metricName, metricFile=metricFile, forceDatasetLoadFolders=datasetNames, detectMemberDataFolders=False)
        Metrics.LoadData.loadMetric(cnnSleepData, metricName=metricName, metricFile=metricFile, forceDatasetLoadFolders=datasetNames, detectMemberDataFolders=False)
        Metrics.LoadData.loadMetric(ffSleepData, metricName=metricName, metricFile=metricFile, forceDatasetLoadFolders=datasetNames, detectMemberDataFolders=False)


        Metrics.Basics.plotSpecificTrialMetricOverDatasetValue(
            [fullSleepData, fullSleepData, cnnSleepData, ffSleepData]
            , datsetNames=datasetNames
            , datsetValues=valueGroup
            , timePoints=[0, 1, 1, 1]
            , metricName=metricName
            , timePointsPrettyNames=["Post Training", "Full Sleep", "Cnn Sleep", "FF Sleep"]
            , prettyXTicks=True
            , prettyFileName=None
            , prettyXLabel=None)


fullSleepData.saveFigures()
cnnSleepData.saveFigures()
ffSleepData.saveFigures()
