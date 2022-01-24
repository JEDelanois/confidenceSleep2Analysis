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


data = SimData(figureFolderPath="../figures/" )


data.createSimulationStructureFromPattern( \
    "../simulations/longTrainMSE/" \
    , "MSE Loss" \
    ,[] \
    , range(0,1)) 

data.createSimulationStructureFromPattern( \
    "../simulations/longTrainBCE/" \
    , "BCE Loss" \
    ,[] \
    , range(0,1)) 

# data.createSimulationStructureSweepFolder( \
    # "../simulationSweep/OLDHiddenClassesSleep/" \
    # , "Hidden Sleep" \
    # , titlePatternSameAsFilePattern=False)

Utils.ConfigUtil.loadConfigsForSimulations(data)

for trainDataset in  [True, False]:

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


    # load all class report, this "dataset" does not have all output file tyeps so hangle it differently
    datasetNames = ["allClassReport"]
    for metricName, metricFile in zip(metricNames, metricFiles):
        Metrics.LoadData.loadMetric(data, metricName=metricName, metricFile=metricFile, forceDatasetLoadFolders=datasetNames, detectMemberDataFolders=False)
        Metrics.Basics.plotTrialMetrics(data, datsetNames=datasetNames, metricNames=[metricName])
        Metrics.Basics.barTrialMetrics(data, datsetNames=datasetNames, metricNames=[metricName], plotIdxs=[0,-1], xticks=["Start Training", "End Training"])

    # load individual class reports
    # datasetNames = ["task1TrainData", "task1TrainDataBlur2", "task1TrainDataBlur10", "task1TrainDataBlur20", "task1TrainDataBlur40"]
    datasetNames = []
    if trainDataset:
        datasetNames = ["task1TrainData", "task1TrainDataBlur2", "task1TrainDataBlur10", "task1TrainDataBlur20", "task1TrainDataBlur40"]
    else:
        datasetNames = ["task1ValidationData", "task1ValidationDataBlur2", "task1ValidationDataBlur10", "task1ValidationDataBlur20", "task1ValidationDataBlur40"]

    metricNames.extend(["loss", "ece"] )
    metricFiles.extend(["loss.txt", "ece.txt"] )

    for metricName, metricFile in zip(metricNames, metricFiles):
        Metrics.LoadData.loadMetric(data, metricName=metricName, metricFile=metricFile, forceDatasetLoadFolders=datasetNames, detectMemberDataFolders=False)

    for metricName in metricNames:
        Metrics.Basics.plotTrialMetrics(data, datsetNames=datasetNames, metricNames=[metricName])
        Metrics.Basics.barTrialMetrics(data, datsetNames=datasetNames, metricNames=[metricName], plotIdxs=[0,-1], xticks=["Start Training", "End Training"])

    # plot all classes for one dataset
    for datasetName in datasetNames:
        for metricSuffix in metricSuffixs:
            filteredMetricNames = [n for n in filter(lambda x: metricSuffix in x, metricNames)]
            # Metrics.Basics.plotTrialMetrics(data, datsetNames=[datasetName], metricNames=filteredMetricNames, prettyFileName="%s-%s-classesAcrossDatasetLine.pdf"%(datasetName, metricSuffix))
            # Metrics.Basics.barTrialMetrics(data, datsetNames=[datasetName], metricNames=filteredMetricNames, plotIdxs=[0,-1], xticks=["Post Training", "Post Sleep"], prettyFileName="%s-%s-classesAcrossDatasetBar.pdf"%(datasetName, metricSuffix))
            Metrics.Basics.plotTrialMetricsDiff(data, datsetNames=[datasetName], metricNames=filteredMetricNames, initialPoint=0, finalPoint=-1, prettyFileName="%s-%s-classesAcrossDatasetPointDiff.pdf"%(datasetName, metricSuffix))

    Confidence.loadMetric(data, datasetFolders=datasetNames)
    Confidence.plotConfidenceHistograms(data, datasetFolders=datasetNames, plotIdxs=[0,-1])


data.saveFigures()