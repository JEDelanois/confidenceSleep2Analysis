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

from simData import *


data = SimData(figureFolderPath="../figures/" )

data.createSimulationStructureFromPattern( \
    "/bazhlab/edelanois/cnnSleep/1/simulations/bestSleep2/" \
    , "Cnn Sleep" \
    ,[] \
    , range(0,1)) 

Utils.ConfigUtil.loadConfigsForSimulations(data)
SleepAnalysis.SleepBasics.loadSleepData(data)
SleepAnalysis.SleepBasics.plotSleepStuff(data)

data.saveFigures()
