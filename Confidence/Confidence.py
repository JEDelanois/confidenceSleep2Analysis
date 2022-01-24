import Utils.GeneralUtil
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import numpy as np

def argmax_first_and_last(a):
    b = torch.stack([torch.arange(a.shape[1])] * a.shape[0])
    max_values, _ = torch.max(a, dim=1)
    b[a != max_values[:, None]] = a.shape[1]
    first_max, _ = torch.min(b, dim=1)
    b[a != max_values[:, None]] = -1
    last_max, _ = torch.max(b, dim=1)
    return first_max, last_max

class ConfObject:
    def __init__(self, networkOutput, labels, epoch):
        self.networkOutput = networkOutput
        self.labels = torch.tensor(labels)
        self.epoch = epoch

def loadMetric(data, datasetFolders=["task1Dataset"]):
    for sim in data.sims:
        for trial in sim.trials:
            trial.data.datasetConfs = {}
            for datasetFolder in datasetFolders:
                trial.data.datasetConfs[datasetFolder] = []
                folderPath = os.path.join(trial.path, datasetFolder, "confidenceData")
                confPaths = [os.path.join(folderPath, fpath) for fpath in os.listdir(folderPath) if fpath.endswith(".pt")]
                confPaths.sort(key=Utils.GeneralUtil.natural_keys)
                for confPath in confPaths:
                    confidenceInfo = torch.load(confPath)
                    trial.data.datasetConfs[datasetFolder].append(ConfObject(epoch=confidenceInfo["epoch"], networkOutput=confidenceInfo["networkOutput"], labels=confidenceInfo["labels"],))

                    # output = trial.data.datasetConfs[datasetFolder][-1].networkOutput

def plotConfidenceHistograms(data, datasetFolders=["task1Dataset"], plotIdxs=None):
    for sim in data.sims:
        for trial in sim.trials:
            for datasetFolder in datasetFolders:
                idxSet = range(len(trial.data.datasetConfs[datasetFolder])) if plotIdxs is None else plotIdxs
                for i in idxSet:
                    sm = nn.Softmax(dim=1)
                    confData = trial.data.datasetConfs[datasetFolder][i]
                    torchNetowrkOutput = torch.tensor(confData.networkOutput)
                    junk, predClasses = argmax_first_and_last(torchNetowrkOutput)
                    softmaxOutput = sm(torchNetowrkOutput)


                    fig, axs = plt.subplots(1, 2)
                    fig.suptitle("%s Conf Distributions | Epoch %d\n" % (datasetFolder, confData.epoch))
                    trial.addFigure(fig, "%s-confDistributions-%d.pdf" % (datasetFolder, confData.epoch))

                    correctIdxs = torch.where(predClasses == confData.labels)[0]
                    correctSoftmax = softmaxOutput[correctIdxs, :]
                    correctPred = predClasses[correctIdxs]
                    correctSoftmaxValues = correctSoftmax[torch.arange(correctIdxs.size()[0]),correctPred].numpy()
                    axs[0].hist(correctSoftmaxValues)
                    axs[0].set_title("Correct confidence distribution\n%d" % (correctSoftmaxValues.shape[0]))

                    incorrectIdxs = torch.where(predClasses != confData.labels)[0]
                    incorrectSoftmax = softmaxOutput[incorrectIdxs, :]
                    incorrectPred = predClasses[incorrectIdxs]
                    incorrectSoftmaxValues = incorrectSoftmax[torch.arange(incorrectIdxs.size()[0]),incorrectPred].numpy()
                    axs[1].hist(incorrectSoftmaxValues)
                    axs[1].set_title("Incorrect confidence distribution\n%d" % (incorrectSoftmaxValues.shape[0]))




            