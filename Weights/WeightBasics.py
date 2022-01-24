import numpy as np
import matplotlib.pyplot as plt
import matplotlib._pylab_helpers
import Weights.WeightUtils 
from sklearn.decomposition import PCA

def plotWeightsOverTime(data):
    for sim in data.sims:
        for trial in sim.trials: 
            for i, layerWeights in enumerate(trial.weights):
                fig = plt.figure()
                unrolledWeights = Weights.WeightUtils.unrollWeightsThroughTime(layerWeights)
                for col in range(unrolledWeights.shape[1]):
                    plt.plot(unrolledWeights[:, col])
                plt.xlabel("Epoch")
                plt.ylabel("Strength")
                plt.title("%s\nLayer %d to %d Weights" % (trial.title, i, i + 1))
                trial.addFigure(fig, ("weightsOverTime%d-%d.pdf" % (i, i + 1)))