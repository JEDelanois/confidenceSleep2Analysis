import numpy as np
import matplotlib.pyplot as plt
import matplotlib._pylab_helpers
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import Weights.WeightUtils 
from sklearn.decomposition import PCA

# data - simulations
# layers - layers to include in weight state ie layers [2,3,4]  - None defaults to all layers
def Pca(data, layers=None, pcaComponents=3):
    for sim in data.sims:
        for trial in sim.trials: 
             # if use all layers
            if layers is None:
                layers = [i for i in range(len(trial.weights))]
            
            pcaWeights = []
            for lay in layers:
                # print("--------------")
                unrolledWeights = Weights.WeightUtils.unrollWeightsThroughTime(trial.weights[lay])
                pcaWeights.append(unrolledWeights)
            
            pcaWeights = np.concatenate(pcaWeights, 1)
            pca = PCA(n_components=pcaComponents)
            pcaPoints = pca.fit_transform(pcaWeights)

            def plot2dPcaProjections(trial, pcaPoints, dim1, dim2, pca):
                fig = plt.figure()
                ax = plt.gca()
                x = pcaPoints[:, dim1]
                y = pcaPoints[:, dim2]
                timeSteps = np.arange(x.shape[0])

                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                # Create a continuous norm to map from data points to colors
                norm = plt.Normalize(timeSteps.min(), timeSteps.max())
                lc = LineCollection(segments, cmap='viridis', norm=norm)
                # Set the values used for colormapping
                lc.set_array(timeSteps)
                lc.set_linewidth(2)
                line = ax.add_collection(lc)
                fig.colorbar(line, ax=ax)
                ax.set_xlim(x.min(), x.max())
                ax.set_ylim(y.min(), y.max())

                # plt.plot(pcaPoints[:, dim1], pcaPoints[:,dim2])
                plt.xlabel("PC %d\nVariance %f" % (dim1, pca.explained_variance_ratio_[dim1]))
                plt.ylabel("PC %d\nVariance %f" % (dim2, pca.explained_variance_ratio_[dim2]))
                plt.title("%s\nPCA ProjectionPC %d - %d" % (trial.title, dim1,dim2))
                plt.tight_layout()
                trial.addFigure(fig, ("pcaProjectionPC%d-%d.png" % (dim1,dim2)))

            plot2dPcaProjections(trial, pcaPoints, 0, 1, pca)
            plot2dPcaProjections(trial, pcaPoints, 0, 2, pca)
            plot2dPcaProjections(trial, pcaPoints, 1, 2, pca)



