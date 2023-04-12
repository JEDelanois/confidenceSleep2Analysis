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


data = SimData(figureFolderPath="../figures/" )

data.createSimulationStructureFromPattern( \
    "../simulations/multiDistortionTest/" \
    , "Cnn Sleep" \
    ,[] \
    , range(0,1)) 

Utils.ConfigUtil.loadConfigsForSimulations(data)
dataGroups = []

# dataGroups.append(["task1ValidData", "task1ValidData-Blur-1", "task1ValidData-Blur-2", "task1ValidData-Blur-3"])
# dataGroups.append(["task1ValidData", "task1ValidData-SP-0_1", "task1ValidData-SP-0_2", "task1ValidData-SP-0_3"])
# dataGroups.append(["task1ValidData", "task1ValidData-GN-0_1", "task1ValidData-GN-0_2", "task1ValidData-GN-0_3"])
# dataGroups.append(["task1ValidData", "task1ValidData-SE-0_25", "task1ValidData-SE-0_50", "task1ValidData-SE-0_75"])
# dataGroups.append(["task1ValidData", "task1ValidData-PN"])

dataGroups.append(["task1ValidData-Blur-1", "task1ValidData-Blur-2", "task1ValidData-Blur-3"])
dataGroups.append(["task1ValidData-SP-0_1", "task1ValidData-SP-0_2", "task1ValidData-SP-0_3"])
dataGroups.append(["task1ValidData-GN-0_1", "task1ValidData-GN-0_2", "task1ValidData-GN-0_3"])
dataGroups.append(["task1ValidData-SE-0_25", "task1ValidData-SE-0_50", "task1ValidData-SE-0_75"])
dataGroups.append(["task1ValidData-PN"])

kernelTypes = ["linear", "poly", "rbf", "sigmoid", "cosine", "precomputed"]

# for dataGroup in dataGroups:
    # for kernelType in kernelTypes:
        # Feature.FeatureMain.getPCASpace(
            # data
            # , modelName="model"
            # , modelNameloadModelPath="../simulations/multiDistortionTest/trial0/stateDict/modelStateDict0.pth"
            # , datasetNames=dataGroup
            # , datasetSymbols=[".", "x", "x", "x"]
            # , outputFolderPrefix="PreSleep"
            # , kernel=kernelType
            # , n_components=3
            # )
        # Feature.FeatureMain.getPCASpace(
            # data
            # , modelName="model"
            # , modelNameloadModelPath="../simulations/multiDistortionTest/trial0/stateDict/modelStateDict1.pth"
            # , datasetNames=dataGroup
            # , datasetSymbols=[".", "x", "x", "x"]
            # , outputFolderPrefix="PostSleep"
            # , kernel=kernelType
            # , n_components=3
            # )

# for dataGroup in dataGroups:
    # Feature.FeatureMain.getKMeansClustering(
        # data
        # , modelName="model"
        # , modelNameloadModelPath="../simulations/multiDistortionTest/trial0/stateDict/modelStateDict0.pth"
        # , datasetNames=dataGroup
        # , datasetSymbols=[".", "x", "x", "x"]
        # , outputFolderPrefix="PreSleep"
        # , n_clusters=10
        # )
    # Feature.FeatureMain.getKMeansClustering(
        # data
        # , modelName="model"
        # , modelNameloadModelPath="../simulations/multiDistortionTest/trial0/stateDict/modelStateDict1.pth"
        # , datasetNames=dataGroup
        # , datasetSymbols=[".", "x", "x", "x"]
        # , outputFolderPrefix="PostSleep"
        # , n_clusters=10
        # )

    # data.saveFigures()

#for dataGroup in dataGroups:
    #Feature.FeatureMain.getSingleDatasetKMeansClustering(
        #data
        #, modelName="model"
        #, modelNameloadModelPath="../simulations/multiDistortionTest/trial0/stateDict/modelStateDict0.pth"
        #, trainDatasetName="task1ValidData"
        #, datasetNames=dataGroup
        #, datasetSymbols=[".", "x", "x", "x"]
        #, outputFolderPrefix="PreSleep"
        #, n_clusters=10
        #)
    #Feature.FeatureMain.getSingleDatasetKMeansClustering(
        #data
        #, modelName="model"
        #, modelNameloadModelPath="../simulations/multiDistortionTest/trial0/stateDict/modelStateDict1.pth"
        #, trainDatasetName="task1ValidData"
        #, datasetNames=dataGroup
        #, datasetSymbols=[".", "x", "x", "x"]
        #, outputFolderPrefix="PostSleep"
        #, n_clusters=10
        #)

    #data.saveFigures()

# look only at train group
# for dataGroup in [["task1TrainData"]]:
    # Feature.FeatureMain.getSingleDatasetKMeansClustering(
        # data
        # , modelName="model"
        # , modelNameloadModelPath="../simulations/multiDistortionTest/trial0/stateDict/modelStateDict0.pth"
        # , trainDatasetName="task1TrainData"
        # , datasetNames=dataGroup
        # , datasetSymbols=[".", "x", "x", "x"]
        # , outputFolderPrefix="PreSleep"
        # , n_clusters=10
        # )
    # Feature.FeatureMain.getSingleDatasetKMeansClustering(
        # data
        # , modelName="model"
        # , modelNameloadModelPath="../simulations/multiDistortionTest/trial0/stateDict/modelStateDict1.pth"
        # , trainDatasetName="task1TrainData"
        # , datasetNames=dataGroup
        # , datasetSymbols=[".", "x", "x", "x"]
        # , outputFolderPrefix="PostSleep"
        # , n_clusters=10
        # )

    # data.saveFigures()
for dataGroup in [["task1TrainData"]]:
    Feature.FeatureMain.getKMeansDiff(
        data
        , modelName="model"
        , modelNameloadModelPath="../simulations/multiDistortionTest/trial0/stateDict/modelStateDict0.pth"
        , trainDatasetNames=["task1TrainData"]
        , datasetNames=dataGroup
        , datasetSymbols=[".", "x", "x", "x"]
        , outputFolderPrefix="PreSleep"
        , n_clusters=10
        )
    Feature.FeatureMain.getKMeansDiff(
        data
        , modelName="model"
        , modelNameloadModelPath="../simulations/multiDistortionTest/trial0/stateDict/modelStateDict1.pth"
        , trainDatasetNames=["task1TrainData"]
        , datasetNames=dataGroup
        , datasetSymbols=[".", "x", "x", "x"]
        , outputFolderPrefix="PostSleep"
        , n_clusters=10
        )

    data.saveFigures()

exit()

# dataGroups.append(["task1TrainData", "task1TrainDataBlur2", "task1TrainDataBlur10", "task1TrainDataBlur20", "task1TrainDataBlur40"]) 
# dataGroups.append(["task1TrainData", "task1TrainDataBlur2", "task1ValidData"]) 
# dataGroups.append(["task1ValidData", "task1ValidDataBlur1", "task1ValidDataBlur1_5", "task1ValidDataBlur2", "task1ValidDataBlur2_5", "task1ValidDataBlur3"])
# dataGroups.append(["task1ValidData", "task1ValidDataSP0-1", "task1ValidDataSP0-25", "task1ValidDataSP0-5"])

# for trainDataset in  [True, False]:
for datasetNames in dataGroups:
    metricNames = ["loss", "matlabAcc"]
    metricFiles = ["loss.txt", "matlabAccuracy.txt"]
    print(datasetNames)
    for metricName, metricFile in zip(metricNames, metricFiles):
        Metrics.LoadData.loadMetric(data, metricName=metricName, metricFile=metricFile, forceDatasetLoadFolders=datasetNames, detectMemberDataFolders=False)
        Metrics.Basics.plotTrialMetrics(data, datsetNames=datasetNames, metricNames=[metricName])

