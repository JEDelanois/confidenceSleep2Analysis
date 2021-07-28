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


# data.createSimulationStructureFromPattern( \
    # "../simulationSweep/HiddenClassesSleep//activationThresholds-[0.01, 0.01, 0.01, 0.5]_weightScales-[1, 5, 10]_delta1-0.0_delta2-0.0_delta3-0.0/".replace(" ","") \
    # , "Sim 1 test" \
    # ,[] \
    # , range(0,1)) 

data.createSimulationStructureSweepFolder( \
    "../simulationSweep/OLDHiddenClassesSleep/" \
    , "Hidden Sleep" \
    , titlePatternSameAsFilePattern=False)

print(len(data.sims))
Utils.ConfigUtil.loadConfigsForSimulations(data)

Metrics.LoadData.loadConfidences(data)
Metrics.LoadData.loadClassMetric2(data)

# DONT USE - this meeses up the order of simulations and can casue big problems
# Metrics.Basics.getRankedConfidences(data, datsetName="task1TrainData", metricName="confidence")

sleepStart = 0
sleepEnd = 1

for datsetName in ["task1TrainData"]:
    for metricName in ["confidence", "classMetric2"]:

        input = [(sim.trials[0].data.datasetMetrics[datsetName][metricName][sleepStart], sim.trials[0].path,) for sim in data.sims]
        Metrics.Basics.printRankedMetric(input, sortReverse=True, sortKey=lambda x: x[0], printNum=10, folderPath=data.figureFolderPath + "/figures/", fileName="%s-first-%s.txt" % (datsetName, metricName))

        input = [(sim.trials[0].data.datasetMetrics[datsetName][metricName][sleepStart], sim.trials[0].path,) for sim in data.sims]
        Metrics.Basics.printRankedMetric(input, sortReverse=True, sortKey=lambda x: x[0], printNum=10, folderPath=data.figureFolderPath + "/figures/", fileName="%s-last-%s.txt" % (datsetName, metricName))

        input = [(sim.trials[0].data.datasetMetrics[datsetName][metricName][sleepEnd]-sim.trials[0].data.datasetMetrics[datsetName][metricName][sleepStart], sim.trials[0].path,) for sim in data.sims]
        Metrics.Basics.printRankedMetric(input, sortReverse=True, sortKey=lambda x: x[0], printNum=10, folderPath=data.figureFolderPath + "/figures/", fileName="%s-diff-%s.txt" % (datsetName, metricName))

        Metrics.Basics.plotMetric(data, datsetName=datsetName, metricName=metricName)

data.saveFigures()



# print(data.sims[0].trials[0].config)
# print(data.sims[0].trials[0].path)

# Vz.VzBasics.loadRewards(data)
# # Vz.VzBasics.plotReward(data, sigma=50)
# Vz.VzBasics.plotAllDataRewards(data)

# data.saveFigures()




    # Utils.RuntimeUtil.reloadLocalModules();Weights.LoadWeightData.loadWeightData(data)
    # Utils.RuntimeUtil.reloadLocalModules();Weights.WeightBasics.plotWeightsOverTime(data)
    # Utils.RuntimeUtil.reloadLocalModules();Weights.Pca.Pca(data)
