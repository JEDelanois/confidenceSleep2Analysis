import torch
import Utils.ConfigUtil
import os
import Utils.GeneralUtil
import Utils.DataUtil
import glob
import code
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib._pylab_helpers
import Weights.WeightUtils 
from sklearn.decomposition import PCA
import matplotlib.colors as colors
from matplotlib.pyplot import cm

def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

def getSortedFilesFromGlob(g):
    folders = glob.glob(g)
    folders.sort(key=natural_keys)
    return folders


# weight shape trial.weights[layer][epoch, post, pre] 
def loadSleepData(data):
    print("Loading Weights")
    for sim in data.sims:
        for trial in sim.trials: 
            sleepFolderPath = trial.path + "/sleepCache/"

            trial.layerNameTolayerIdx = torch.load(trial.path + "/sleepCache/layerNameToLayerIdx.pt")
            trial.layerIdxToLayerName = torch.load(trial.path + "/sleepCache/layerIdxToLayerName.pt")
            trial.layerNameTolayerIdx["inputSpikes"] = -1 

            # Map layer name to (curTimeStep, postSpikes)
            trial.spikes = {}
            trial.sumSpikes = {}
            trial.voltages = {}
            trial.voltagesPreReset = {}
            trial.preWeights = {}
            trial.postWeights = {}

            print("Loading spikes/potentials/... for simulation %s" % trial.path)
            for layerName in trial.layerNameTolayerIdx.keys():
                path = sleepFolderPath + "/Layer={}.txt".format(layerName)
                if os.path.exists(path):
                    trial.sumSpikes[layerName] = Utils.DataUtil.numpyDataQuickLoadEasy(path)

                files = getSortedFilesFromGlob("%s/membranePotentials/layerForward_iter=*_layer=%s.pt" % (sleepFolderPath, layerName))
                if len(files) > 0:
                    trial.voltages[layerName] = []
                    for file in files: 
                        trial.voltages[layerName].append(torch.load(file))
                    trial.voltages[layerName] = np.concatenate(trial.voltages[layerName], axis=0)

                files = getSortedFilesFromGlob("%s/membranePotentialsBeforeReset/layerForward_iter=*_layer=%s.pt" % (sleepFolderPath, layerName))
                if len(files) > 0:
                    trial.voltagesPreReset[layerName] = []
                    for file in files: 
                        trial.voltagesPreReset[layerName].append(torch.load(file))
                    trial.voltagesPreReset[layerName] = np.concatenate(trial.voltagesPreReset[layerName], axis=0)
                
                
                files = getSortedFilesFromGlob("%s/spikeOutputsCache/layerForward_iter=*_layer=%s.pt" % (sleepFolderPath, layerName))
                if len(files) > 0:
                    trial.spikes[layerName] = []
                    for file in files: 
                        trial.spikes[layerName].append(torch.load(file))
                    trial.spikes[layerName] = np.concatenate(trial.spikes[layerName], axis=0)

                # files = getSortedFilesFromGlob("%s/preWeights/layerForward_iter=*_layer=%s.pt" % (sleepFolderPath, layerName))
                # if len(files) > 0:
                    # trial.preWeights[layerName] = []
                    # for file in files: 
                        # trial.preWeights[layerName] = torch.load(file)

                # files = getSortedFilesFromGlob("%s/postWeights/layerForward_iter=*_layer=%s.pt" % (sleepFolderPath, layerName))
                # if len(files) > 0:
                    # trial.postWeights[layerName] = []
                    # for file in files: 
                        # trial.postWeights[layerName] = torch.load(file)

                files = getSortedFilesFromGlob("%s/preWeights/unitForward_layer=%s.pt" % (sleepFolderPath, layerName))
                if len(files) > 0:
                    file = files[0]
                    trial.preWeights[layerName] = torch.load(file)

                files = getSortedFilesFromGlob("%s/postWeights/unitForward_layer=%s.pt" % (sleepFolderPath, layerName))
                if len(files) > 0:
                    file = files[0]
                    trial.postWeights[layerName] = torch.load(file)

def combine_dims(a, start=1, count=2):
    """ Reshapes numpy array a by combining count dimensions, 
        starting at dimension index start """
    s = a.shape
    return np.reshape(a, s[:start] + (-1,) + s[start+count:])

def plotSleepStuff(data):
    for sim in data.sims:
        for trial in sim.trials: 

            fig = plt.figure()
            sortedKeys = [key for key in trial.sumSpikes]
            sortedKeys.sort(key=natural_keys)
            sortedKeys.insert(0,sortedKeys.pop()) # currently last key is input spikes when we want first key to be input spikes
            color = cm.cool(np.linspace(0, 1, len(sortedKeys)))
#TODO need these to be sorted in forward pass order starting with input layer
            for i,key in enumerate(sortedKeys):
                plt.plot(trial.sumSpikes[key][:,0], trial.sumSpikes[key][:,1], c=color[i], )
            # plt.legend(sortedKeys, loc=(1.04,0))
            plt.yscale("log")
            plt.ylabel("Sum Spikes")
            plt.xlabel("Sleep Iteration")
            plt.tight_layout()
            trial.addFigure(fig, "allSumSpikes.png")

            for layerName in trial.spikes:
                fig = plt.figure()
                plt.plot(trial.sumSpikes[layerName][:,0], trial.sumSpikes[layerName][:,1])
                plt.title("SumSpikes for %s" % layerName)
                plt.xlabel("Iteration Number")
                plt.ylabel("Number of Spikes")
                trial.addFigure(fig, "sumSpikes-%s.png" % (layerName))

                fig = plt.figure()
                numDims = len(trial.spikes[layerName].shape)
                reshaped = combine_dims(trial.spikes[layerName], start=1, count=numDims-1)
                plt.imshow(reshaped, aspect="auto", cmap="jet", interpolation='none')
                plt.title("Spikes for %s" % layerName)
                plt.xlabel("Feature Idx")
                plt.ylabel("Iteration Number")
                plt.colorbar()
                trial.addFigure(fig, "spikes-%s.png" % (layerName))

            for layerName in trial.voltages:
                fig = plt.figure()
                numDims = len(trial.voltages[layerName].shape)
                reshaped = combine_dims(trial.voltages[layerName], start=1, count=numDims-1)
                plt.imshow(reshaped, aspect="auto", cmap="jet", interpolation='none')
                plt.title("Voltage for %s" % layerName)
                plt.xlabel("Feature Idx")
                plt.ylabel("Iteration Number")
                plt.colorbar()
                trial.addFigure(fig, "voltages-%s.png" % (layerName))

            for layerName in trial.voltagesPreReset:
                fig = plt.figure()
                numDims = len(trial.voltagesPreReset[layerName].shape)
                reshaped = combine_dims(trial.voltagesPreReset[layerName], start=1, count=numDims-1)
                plt.imshow(reshaped, aspect="auto", cmap="jet", interpolation='none',  norm=colors.Normalize(vmin=trial.voltagesPreReset[layerName].min(), vmax=trial.voltagesPreReset[layerName].max()))
                plt.title("Voltage for %s" % layerName)
                plt.xlabel("Feature Idx")
                plt.ylabel("Iteration Number")
                plt.colorbar()
                trial.addFigure(fig, "PreResetVoltages-%s.png" % (layerName))

            # for layerName in trial.preWeights:
            for layerName in trial.postWeights:
                fig = plt.figure()
                pre = trial.preWeights[layerName]
                post = trial.postWeights[layerName]
                plt.hist(post.ravel() - pre.ravel())
                plt.title("Weight Diff for Layer %s" % layerName)
                plt.xlabel("Magnitue of Weight Differential")
                plt.ylabel("Number of weights")
                trial.addFigure(fig, "weightDiffHist-%s.png" % (layerName))

