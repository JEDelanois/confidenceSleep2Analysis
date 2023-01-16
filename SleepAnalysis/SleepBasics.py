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
import code
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

def plotSleepStuff(data, plotFilters=False):
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

            # for layerName in trial.preWeights:
            for layerName in trial.postWeights:
                fig = plt.figure()
                pre = trial.preWeights[layerName]
                post = trial.postWeights[layerName]
                plt.hist(post.ravel() - pre.ravel())
                plt.title("Weight Diff for Layer %s" % layerName)
                plt.xlabel("Magnitue of Weight Differential")
                plt.ylabel("Number of weights")
                plt.yscale("log")
                trial.addFigure(fig, "weightDiffHist-%s.png" % (layerName))

            for layerName in trial.postWeights:
                fig = plt.figure()
                wts = trial.postWeights[layerName]
                plt.hist(wts.ravel())
                plt.title("Weight Post Sleep %s" % layerName)
                plt.xlabel("Magnitue of Weight")
                plt.ylabel("Number of weights")
                trial.addFigure(fig, "weightPostSleepHist-%s.png" % (layerName))

            if plotFilters:
                for layerName in trial.preWeights:
                    pre = trial.preWeights[layerName]
                    if len(pre.shape) > 2: # assume it is a conv filter
                        fig, axs = plt.subplots(pre.shape[0], pre.shape[1])

                        # need to modify shapes to make generalizable when dimmension is 1
                        if pre.shape[0] == 1:
                            axs = np.expand_dims(axs, 0)
                        if pre.shape[1] == 1:
                            axs = np.expand_dims(axs, 1)

                        minVal = pre.min()
                        maxVal = pre.max()
                        if minVal == 0.0 and maxVal == 0.0:
                            minVal = -0.0001
                            maxVal = 0.0001
                        for i in range(pre.shape[0]):
                            for j in range(pre.shape[1]):
                                im = axs[i][j].imshow(pre[i,j,:,:], aspect='auto', interpolation='none', vmin=minVal, vmax=maxVal)
                                # fig.colorbar(im, cax=axs[i][j])
                                axs[i][j].axes.xaxis.set_visible(False)
                                axs[i][j].axes.yaxis.set_visible(False)
                                axs[i][j].axis('equal')
                                axs[i][j].spines['top'].set_visible(False)
                                axs[i][j].spines['right'].set_visible(False)
                                axs[i][j].spines['bottom'].set_visible(False)
                                axs[i][j].spines['left'].set_visible(False)
                        fig.suptitle("Presleep weight visualizations %s" % layerName)
                        fig.colorbar(im, ax=axs.ravel().tolist())
                        # fig.tight_layout()
                        trial.addFigure(fig, "weightVisualizations/preSleep-%s.png" % (layerName))
                    else:
                        fig = plt.figure()
                        plt.imshow(pre, aspect='auto', interpolation='none')
                        plt.title("Presleep weight visualizations %s" % layerName)
                        plt.colorbar()
                        trial.addFigure(fig, "weightVisualizations/preSleep-%s.png" % (layerName))

                for layerName in trial.preWeights:
                    pre = trial.preWeights[layerName]
                    post = trial.postWeights[layerName]
                    if len(pre.shape) > 2: # assume it is a conv filter

                        # need to modify shapes to make generalizable when dimmension is 1
                        if pre.shape[0] == 1:
                            axs = np.expand_dims(axs, 0)
                        if pre.shape[1] == 1:
                            axs = np.expand_dims(axs, 1)

                        minVal = min(pre.min(), post.min())
                        maxVal = max(pre.max(), post.max())
                        # make symetric around 0
                        lim = max(abs(minVal), abs(maxVal))
                        minVal = -lim
                        maxVal = lim
                        if minVal == 0.0 and maxVal == 0.0:
                            minVal = -0.0001
                            maxVal = 0.0001
                        for i in range(pre.shape[0]):
                            for j in range(pre.shape[1]):
                                fig, axs = plt.subplots(1, 3)
                                fig.suptitle("filter-%d-%d" % (i,j))
                                try:
                                    # set plotwise min and max vals
                                    # if you want global min / max vals then 
                                    minVal = min(pre[i,j,:,:].min(), post[i,j,:,:].min())
                                    maxVal = min(pre[i,j,:,:].max(), post[i,j,:,:].max())
                                    lim = max(abs(minVal), abs(maxVal))
                                    minVal = -lim
                                    maxVal = lim
                                    if minVal == 0.0 and maxVal == 0.0:
                                        minVal = -0.0001
                                        maxVal = 0.0001

                                    im = axs[0].imshow(pre[i,j,:,:], aspect='auto', interpolation='none', vmin=minVal, vmax=maxVal, cmap="coolwarm")
                                    # im = axs[0].imshow(pre[i,j,:,:], aspect='auto', interpolation='none', norm=colors.CenteredNorm(), cmap="coolwarm")
                                    # im = axs[0].pcolormesh(pre[i,j,:,:], norm=colors.CenteredNorm(), cmap="coolwarm")
                                    divider = make_axes_locatable(axs[0])
                                    cax = divider.append_axes('right', size='5%', pad=0.05)
                                    fig.colorbar(im, cax=cax, orientation='vertical')
                                    axs[0].set_title("PreSleep Weights")

                                    im = axs[1].imshow(post[i,j,:,:], aspect='auto', interpolation='none', vmin=minVal, vmax=maxVal, cmap="coolwarm")
                                    # im = axs[1].imshow(post[i,j,:,:], aspect='auto', interpolation='none', norm=colors.CenteredNorm(), cmap="coolwarm")
                                    divider = make_axes_locatable(axs[1])
                                    cax = divider.append_axes('right', size='5%', pad=0.05)
                                    fig.colorbar(im, cax=cax, orientation='vertical')
                                    axs[1].set_title("PostSleep Weights")
                                
                                    diff = post[i,j,:,:] - pre[i,j,:,:]
                                    lim = max(abs(diff.min()), abs(diff.max()))
                                    im = axs[2].imshow(diff, aspect='auto', interpolation='none', vmin=-lim, vmax=lim, cmap="coolwarm")
                                    # im = axs[2].imshow(diff, aspect='auto', interpolation='none', norm=colors.CenteredNorm(), cmap="coolwarm")
                                    divider = make_axes_locatable(axs[2])
                                    cax = divider.append_axes('right', size='5%', pad=0.05)
                                    fig.colorbar(im, cax=cax, orientation='vertical')
                                    axs[2].set_title("Difference")
                                except Exception as e:
                                    print(e)
                                    code.interact(local=dict(globals(), **locals()))

                                # fig.colorbar(im, cax=axs[i][j])
                                for jj in range(3):
                                    axs[jj].axes.xaxis.set_visible(False)
                                    axs[jj].axes.yaxis.set_visible(False)
                                    axs[jj].axis('equal')
                                    axs[jj].spines['top'].set_visible(False)
                                    axs[jj].spines['right'].set_visible(False)
                                    axs[jj].spines['bottom'].set_visible(False)
                                    axs[jj].spines['left'].set_visible(False)
                                trial.addFigure(fig, "weightVisualizations/layerWeights-%s_all/filter-%d-%d.png" % (layerName,i,j))
                        # fig.tight_layout()

                for layerName in trial.preWeights:
                    try:
                        pre = trial.preWeights[layerName].ravel()
                        post = trial.postWeights[layerName].ravel()
                        diff = post - pre
                        fig = plt.figure()
                        plt.scatter(pre, diff)
                        plt.xlabel("PreSleep weight magniture")
                        plt.ylabel("Sleep Delta")
                        plt.title(layerName)
                        trial.addFigure(fig, "weightVisualizations/%s-differentialvsInitialWeight.png" % (layerName))
                    except Exception as e:
                        print(e)
                        code.interact(local=dict(globals(), **locals()))

                for layerName in trial.postWeights:
                    post = trial.postWeights[layerName]
                    if len(post.shape) > 2: # assume it is a conv filter
                        fig, axs = plt.subplots(post.shape[0], post.shape[1])

                        # need to modify shapes to make generalizable when dimmension is 1
                        if post.shape[0] == 1:
                            axs = np.expand_dims(axs, 0)
                        if post.shape[1] == 1:
                            axs = np.expand_dims(axs, 1)

                        minVal = post.min()
                        maxVal = post.max()
                        if minVal == 0.0 and maxVal == 0.0:
                            minVal = -0.0001
                            maxVal = 0.0001
                        for i in range(post.shape[0]):
                            for j in range(post.shape[1]):
                                im = axs[i][j].imshow(post[i,j,:,:], aspect='auto', interpolation='none', vmin=minVal, vmax=maxVal)
                                # fig.colorbar(im, cax=axs[i][j])
                                axs[i][j].axes.xaxis.set_visible(False)
                                axs[i][j].axes.yaxis.set_visible(False)
                                axs[i][j].axis('equal')
                                axs[i][j].spines['top'].set_visible(False)
                                axs[i][j].spines['right'].set_visible(False)
                                axs[i][j].spines['bottom'].set_visible(False)
                                axs[i][j].spines['left'].set_visible(False)
                        fig.suptitle("Postsleep weight visualizations %s" % layerName)
                        fig.colorbar(im, ax=axs.ravel().tolist())
                        trial.addFigure(fig, "weightVisualizations/postSleep-%s.png" % (layerName))
                    else:
                        fig = plt.figure()
                        plt.imshow(post, aspect='auto', interpolation='none')
                        plt.title("Postsleep weight visualizations %s" % layerName)
                        plt.colorbar()
                        trial.addFigure(fig, "weightVisualizations/postSleep-%s.png" % (layerName))

                for layerName in trial.postWeights:
                    pre = trial.preWeights[layerName]
                    post = trial.postWeights[layerName]
                    diff = post-pre
                    if len(diff.shape) > 2: # assume it is a conv filter
                        fig, axs = plt.subplots(diff.shape[0], diff.shape[1])

                        # need to modify shapes to make generalizable when dimmension is 1
                        if diff.shape[0] == 1:
                            axs = np.expand_dims(axs, 0)
                        if diff.shape[1] == 1:
                            axs = np.expand_dims(axs, 1)

                        minVal = diff.min()
                        maxVal = diff.max()

                        if minVal == 0 and maxVal == 0:
                            minVal = -0.0001
                            maxVal = 0.0001

                        # code.interact(local=dict(globals(), **locals()))
                        for i in range(diff.shape[0]):
                            for j in range(diff.shape[1]):
                                im = axs[i][j].imshow(diff[i,j,:,:], aspect='auto', interpolation='none', vmin=minVal, vmax=maxVal)
                                # fig.colorbar(im, cax=axs[i][j])
                                axs[i][j].axes.xaxis.set_visible(False)
                                axs[i][j].axes.yaxis.set_visible(False)
                                axs[i][j].axis('equal')
                                axs[i][j].spines['top'].set_visible(False)
                                axs[i][j].spines['right'].set_visible(False)
                                axs[i][j].spines['bottom'].set_visible(False)
                                axs[i][j].spines['left'].set_visible(False)
                        fig.suptitle("Diff sleep weight visualizations %s" % layerName)
                        fig.colorbar(im, ax=axs.ravel().tolist())
                        trial.addFigure(fig, "weightVisualizations/diffSleep-%s.png" % (layerName))
                    else:
                        fig = plt.figure()
                        plt.imshow(diff, aspect='auto', interpolation='none')
                        plt.title("diff sleep weight visualizations %s" % layerName)
                        plt.colorbar()
                        trial.addFigure(fig, "weightVisualizations/diffSleep-%s.png" % (layerName))
