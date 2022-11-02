import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import Utils.ConfigUtil as ConfigUtil
sys.path.append("../")
sys.path.append("../../")
from Simulation import *
import code
import copy



def getConvLayerWeiths(model):
    return 


def reshapeConvFilters(convBlock):
    w = convBlock.weight.detach().clone().numpy()
    filterSize = w.shape[2] * w.shape[3]
    numberFilters = w.shape[0] * w.shape[1]
    allFilters = []
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            allFilters.append(w[i,j,:,:].reshape(filterSize))
    return np.vstack(tuple(allFilters))
    # return w.reshape(numberFilters, filterSize) # this reshape should interate over last index first so 


def sparsity(filt):
    return

def orthogonality(filt): # for this you will need to remove sparse filters and the paper only focuses on cin blocks, not entire layer
    # # remove sparse filters
    m = float(np.max(np.abs(filt)))
    notSparseIdxs = np.where(np.max(np.abs(filt), axis=1) >= (m / 2.5)) # get all filters that have a max value of less than a fraction of the max
    print(notSparseIdxs)
    filt = filt[notSparseIdxs]

    norms = np.linalg.norm(filt, axis=1, ord=2)
    filt = filt / norms[:, None] # normalize to unit length
    prod = np.matmul(filt, filt.T) - np.identity(filt.shape[0], dtype=float)
    normProd = np.linalg.norm(prod, ord=1)
    # normProd = np.sum(prod)
    frac = normProd / (float(filt.shape[0]) * float(filt.shape[0] - 1))
    return 1.0 - frac

def sparsity(filt):
    m = float(np.max(np.abs(filt)))
    counts = np.where(np.max(np.abs(filt), axis=1) <= (m / 5.0), 1, 0) # get all filters that have a max value of less than a fraction of the max
    ret = float(np.sum(counts)) / float(counts.shape[0])
    # code.interact(local=dict(globals(), **locals()))
    return ret

def getFilterMean(filt):
    return np.mean(filt)

def getFilterStd(filt):
    return np.std(filt)

def testOrthogonality():
    orth = np.identity(9, dtype=float)
    x = np.array([1., 0., 0., 0., 0., 0., 0., 0., 0.])
    y = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
    parallel = np.vstack([x,x,x,x,x,x,x,x,x,x,x,x,x])
    mix = np.vstack([y,x,x,x,x,x,x,x,x,x,x,x,x])
    print("orthogonality of orth")
    print(orthogonality(orth))
    print("orthogonality of parallel")
    print(orthogonality(parallel))
    print("orthogonality of mixed")
    print(orthogonality(mix))


def getConvBlocks(model):
    if type(model) is SmallCNN:
        return [model.conv1, model.conv2]

# y axis - metric value
# x axis - filt depth
# line color - time point
def plotMetric(datas, timePointPaths, fileName="filter-Orthogonality.png", metricType="orthogonality", modelMemberName="model", legend=["Post Training", "Post Sleep"]):
    fig = plt.figure()
    plt.title(metricType)
    for i, data in enumerate(datas):
        vals = []
        data.addFigure(fig, fileName)
        for trial in data.sims[0].trials:
            modelConfig = copy.deepcopy(ConfigUtil.getMemberSetWithName(trial.config, modelMemberName))
            modelConfig[2]["preloadParamsFromFile"] = trial.path + timePointPaths[i]
            model = SYMBOLS[modelConfig[1]].generateFromConfig(modelConfig[2])
            convBlocks = getConvBlocks(model)
            filters = [reshapeConvFilters(convBlock) for convBlock in convBlocks]
            metricFunc = globals()[metricType]
            values = [metricFunc(f) for f in filters]
            vals.append(values)
        vals = np.array(vals)
        means = np.mean(vals, axis=0)
        stds = np.std(vals, axis=0)
        # plt.plot(means, label=legend[i])
        plt.scatter(np.arange(means.shape[0]), means, label=legend[i])
    plt.xlabel("Conv Layer")
    plt.ylabel(metricType)
    plt.legend()
    # code.interact(local=dict(globals(), **locals()))
