import matplotlib
# matplotlib.use("agg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib._pylab_helpers

import Rewards.LoadRewards

import Weights.LoadWeightData 
import Weights.Pca
import Weights.WeightBasics
import Utils.ConfigUtil
import Utils.RuntimeUtil

from simData import *

data = SimData()

# data.createSimulationStructureFromPattern("../pop{0}/hidNeurons{1}/",[[1],[100, 200, 300, 400, 500, 600, 700, 750]], range(1,11))
# data.createSimulationStructureFromPattern( \
    # "../simulations/interleavedTraining/amBpInterleaved" \
    # , "AmBp | Train 1 | Interleaved Sleep | " \
    # ,[] \
    # , range(0,3)) 

# data.createSimulationStructureFromPattern( \
    # "../simulations/interleavedTraining/BpInterleaved" \
    # , "Bp | Train 1 | Interleaved Sleep | " \
    # ,[] \
    # , range(0,3)) 

data.createSimulationStructureFromPattern( \
    "../simulations/sim1" \
    , "Sim 1 test" \
    ,[] \
    , range(0,1)) 

Utils.ConfigUtil.loadConfigsForSimulations(data)

Rewards.LoadRewards.loadRewards(data)


# print(data.sims[0].trials[0].config)
print(data.sims[0].trials[0].path)
# Utils.RuntimeUtil.reloadLocalModules();Weights.LoadWeightData.loadWeightData(data)
# Utils.RuntimeUtil.reloadLocalModules();Weights.WeightBasics.plotWeightsOverTime(data)
# Utils.RuntimeUtil.reloadLocalModules();Weights.Pca.Pca(data)

plt.show()
data.saveFigures()