import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
sys.path.append("../")
sys.path.append("../../")
from Simulation import *
import code
from torch.utils.data import DataLoader
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from sklearn.decomposition import KernelPCA
from sklearn.cluster import KMeans
from sklearn import metrics
import copy

def getAllDataFromDataset(
    dataset
    , dataLoaderParams = {
        "shuffle":False
        , "batch_size":10
        , "num_workers":15
        , "pin_memory":True
        }
    ):

    allData = []
    dataLoaderParams["batch_size"] = len(dataset)
    dl = DataLoader(dataset , **dataLoaderParams) # convert datasets into dataloader instances
    for batchData, batchLabels in dl:
        pass
    return batchData, batchLabels

def getActivationsForDatasets(model, datasets, datasetSymbols=None, flatten=True):
    allActivations = [] # allActivations[layerIdx][exampleIdx] = features
    allLabels = [] # allLabels[exampleIdx] = label
    markers = [] # marker[exampleIdx] = marker
    flattenIdx = 1 if flatten else -1
    for datasetIdx, dataset in enumerate(datasets):
        model(dataset[0])
        allLabels.extend(torch.argmax(dataset[1], dim=1).tolist())
        if datasetSymbols is not None:
            markers.extend([datasetSymbols[datasetIdx]] * dataset[1].shape[0])
        # print(len(allLabels))
        # print(len(markers))

        for i,activation in enumerate(model.activations):
            if i >= len(allActivations):
                allActivations.append(torch.tensor([]))
            allActivations[i] = torch.cat([allActivations[i], model.activations[i].flatten(start_dim=flattenIdx).detach()], dim=0) # flattens all features to single dimmension

    return allActivations, torch.tensor(allLabels), markers 

# def getFlattenedActivationsForDatasets(model, datasets, datasetSymbols=None):
    # allActivations = [] # allActivations[layerIdx][exampleIdx] = features
    # allLabels = [] # allLabels[exampleIdx] = label
    # markers = [] # marker[exampleIdx] = marker
    # for datasetIdx, dataset in enumerate(datasets):
        # model(dataset[0])
        # allLabels.extend(torch.argmax(dataset[1], dim=1).tolist())
        # if datasetSymbols is not None:
            # markers.extend([datasetSymbols[datasetIdx]] * dataset[1].shape[0])
        # # print(len(allLabels))
        # # print(len(markers))

        # for i,activation in enumerate(model.activations):
            # if i >= len(allActivations):
                # allActivations.append(torch.tensor([]))
            # allActivations[i] = torch.cat([allActivations[i], model.activations[i].flatten(start_dim=1).detach()], dim=0) # flattens all features to single dimmension

    # return allActivations, torch.tensor(allLabels), markers 

def getPCASpace(
    data
    , modelName="model"
    , modelNameloadModelPath="../simulations/multiDistortionTest/trial0/stateDict/modelStateDict0.pth"
    , datasetNames=["task1ValidData", "task1ValidData-Blur-1", "task1ValidData-Blur-2", "task1ValidData-Blur-3"]
    , datasetSymbols=[".", "x", "x", "x"]
    , outputFolderPrefix="PreSleep"
    , kernel="linear"
    , n_components=3
    ):


    for sim in data.sims:
        for trial in sim.trials: 
            simObj = Simulation(trial.config, **trial.config)
            simObj.createSimMembers(trial.config["members"], trial.config["modifiers"], trial.config["stages"], cachedSimMembers={})
            model = simObj.members[modelName]
            model.load_state_dict(torch.load(modelNameloadModelPath, map_location=devices.comp))
            datasets = [simObj.members[d] for d in datasetNames]
            datasets = [getAllDataFromDataset(d) for d in datasets]


            allActivations, allLabels, markers  = getActivationsForDatasets(model, datasets, datasetSymbols)

            def plotPcScatter(features, pca, labels, markers, dim0, dim1, folderPrefix="layer0"):
                colors = [key for key in mcolors.TABLEAU_COLORS]
                fig = plt.figure()
                trial.addFigure(fig, "/features/%s/%s-pca-%d-%d.png" % (folderPrefix, str(datasetNames), dim0, dim1))
                points = pca.transform(features)
                plotcolors = [colors[v] for v in labels]
                # plt.scatter(points[:,dim0], points[:,dim1], c=plotcolors, marker=markers)
                plt.scatter(points[:,dim0], points[:,dim1], c=plotcolors)
                # plt.title("Dim %d - %f Variance | Dim %d - %f Variance" % (dim0, pca.explained_variance_ratio_[dim0], dim1, pca.explained_variance_ratio_[dim1])) # explained variance makes sense with respect to standard pca, not KPCA
                plt.title("Dim %d | Dim %d" % (dim0, dim1))
                # plt.legend([str(i) for i in range(colors)])
                plt.legend(handles=[mpatches.Patch(color=cc, label=str(i)) for i,cc in enumerate(colors)])
                plt.xlabel("PC %d" % dim0)
                plt.ylabel("PC %d" % dim1)

            for i,layerActivations in enumerate(allActivations):
                # print("transforming")
                # pca = PCA(n_components=n_components)
                pca = KernelPCA(n_components=7, kernel=kernel)
                pca.fit(layerActivations)
                plotPcScatter(layerActivations, pca, allLabels, markers, 0, 1, folderPrefix="/%s/%s/layer%d" % (outputFolderPrefix,kernel,i))
                plotPcScatter(layerActivations, pca, allLabels, markers, 0, 2, folderPrefix="/%s/%s/layer%d" % (outputFolderPrefix,kernel,i))
                plotPcScatter(layerActivations, pca, allLabels, markers, 1, 2, folderPrefix="/%s/%s/layer%d" % (outputFolderPrefix,kernel,i))
                # code.interact(local=dict(globals(), **locals()))


            # model = SmallCNN(**config)
            # if config["preloadParamsFromFile"] is not None: # if want to use custom pretrained model then do so
                # model.load_state_dict(torch.load(config["preloadParamsFromFile"], map_location=devices.comp))

def getKMeansClustering(
    data
    , modelName="model"
    , modelNameloadModelPath="../simulations/multiDistortionTest/trial0/stateDict/modelStateDict0.pth"
    , datasetNames=["task1ValidData", "task1ValidData-Blur-1", "task1ValidData-Blur-2", "task1ValidData-Blur-3"]
    , datasetSymbols=[".", "x", "x", "x"]
    , outputFolderPrefix="PreSleep"
    , n_clusters=10
    ):


    for sim in data.sims:
        for trial in sim.trials: 
            simObj = Simulation(trial.config, **trial.config)
            simObj.createSimMembers(trial.config["members"], trial.config["modifiers"], trial.config["stages"], cachedSimMembers={})
            model = simObj.members[modelName]
            model.load_state_dict(torch.load(modelNameloadModelPath, map_location=devices.comp))
            datasets = [simObj.members[d] for d in datasetNames]
            datasets = [getAllDataFromDataset(d) for d in datasets]


            allActivations, allLabels, markers  = getActivationsForDatasets(model, datasets, datasetSymbols)

            def plotKmeansStuff(features, kmeans, labels, markers, folderPrefix="layer0"):
                colors = [key for key in mcolors.TABLEAU_COLORS]

                allDists = kmeans.transform(features)
                clusterIds = kmeans.predict(features)
                distToCluster =  allDists[[i for i in range(allDists.shape[0])],clusterIds]

                clusterDists = []
                plotLabels = []
                for i in range(n_clusters):
                    plotLabels.append(str(i))
                    idxs = np.where(clusterIds == i)
                    dist = distToCluster[idxs]
                    clusterDists.append(dist)
                clusterDists.append(distToCluster)
                plotLabels.append(str("All Groups"))

                fig = plt.figure()
                plt.boxplot(clusterDists, meanline=True, labels=plotLabels)
                trial.addFigure(fig, "/features/kmeans/%s/%s-boxplot.png" % (folderPrefix, str(datasetNames)))

                # code.interact(local=dict(globals(), **locals()))

            def plotKmeansClassifications(features, kmeans, labels, markers, folderPrefix="layer0"):
                colors = [key for key in mcolors.TABLEAU_COLORS]

                clusterIds = kmeans.predict(features)
                cm = metrics.confusion_matrix(labels, clusterIds, normalize="true")

                fig = plt.figure()
                trial.addFigure(fig, "/features/kmeans/%s/%s-CMs.png" % (folderPrefix, str(datasetNames)))
                plt.imshow(cm, interpolation='none', cmap="jet", aspect="auto", vmin=0, vmax=1.0)
                plt.ylabel("True Label")
                plt.xlabel("Kmeans Cluster Id")
                plt.colorbar()
                ax = plt.gca()

                for (j,i),label in np.ndenumerate(cm):
                    label = "%.2f" % label
                    ax.text(i,j,label,ha='center',va='center')
                    ax.text(i,j,label,ha='center',va='center')

            for i,layerActivations in enumerate(allActivations):
                kmeans = KMeans(n_clusters=n_clusters, random_state=0)
                kmeans.fit(layerActivations)
                # kmeans.cluster_centers_
                plotKmeansStuff(layerActivations, kmeans, allLabels, markers, folderPrefix="/%s/layer%d" % (outputFolderPrefix,i))
                plotKmeansClassifications(layerActivations, kmeans, allLabels, markers, folderPrefix="/%s/layer%d" % (outputFolderPrefix,i))
                # code.interact(local=dict(globals(), **locals()))

# TODO 
# get difference between distance to k means centriods trained on task1ValidData only and plot their distribution
# make sure that clusters correspond to actualy mnist classes
# look at change in confusion matrixes for clustoring algorithm
def getSingleDatasetKMeansClustering(
    data
    , modelName="model"
    , modelNameloadModelPath="../simulations/multiDistortionTest/trial0/stateDict/modelStateDict0.pth"
    , trainDatasetName="task1ValidData"
    , datasetNames=["task1ValidData-Blur-1", "task1ValidData-Blur-2", "task1ValidData-Blur-3"]
    , datasetSymbols=[".", "x", "x", "x"]
    , outputFolderPrefix="PreSleep"
    , n_clusters=10
    ):


    for sim in data.sims:
        for trial in sim.trials: 
            simObj = Simulation(trial.config, **trial.config)
            simObj.createSimMembers(trial.config["members"], trial.config["modifiers"], trial.config["stages"], cachedSimMembers={})
            model = simObj.members[modelName]
            model.load_state_dict(torch.load(modelNameloadModelPath, map_location=devices.comp))
            datasets = [simObj.members[d] for d in datasetNames]
            datasets = [getAllDataFromDataset(d) for d in datasets]

            trainDataset = simObj.members[trainDatasetName]
            trainDataset = getAllDataFromDataset(trainDataset)

            trainActivations , trainLabels , trainMarkers  = getActivationsForDatasets(model, [trainDataset], datasetSymbols) 
            allActivations, allLabels, markers  = getActivationsForDatasets(model, datasets, datasetSymbols)

            def plotKmeansStuff(features, kmeans, labels, markers, folderPrefix="layer0"):
                colors = [key for key in mcolors.TABLEAU_COLORS]

                allDists = kmeans.transform(features)
                clusterIds = kmeans.predict(features)
                distToCluster =  allDists[[i for i in range(allDists.shape[0])],clusterIds]

                clusterDists = []
                plotLabels = []
                for i in range(n_clusters):
                    plotLabels.append(str(i))
                    idxs = np.where(clusterIds == i)
                    dist = distToCluster[idxs]
                    clusterDists.append(dist)
                clusterDists.append(distToCluster)
                plotLabels.append(str("All Groups"))

                fig = plt.figure()
                plt.boxplot(clusterDists, meanline=True, labels=plotLabels)
                trial.addFigure(fig, "/features/dataset%s_Kmeans/%s/%s-boxplot.png" % (trainDatasetName,folderPrefix, str(datasetNames)))

                # code.interact(local=dict(globals(), **locals()))

            def plotKmeansClassifications(features, kmeans, labels, markers, folderPrefix="layer0"):
                colors = [key for key in mcolors.TABLEAU_COLORS]

                clusterIds = kmeans.predict(features)
                cm = metrics.confusion_matrix(labels, clusterIds, normalize="true")

                fig = plt.figure()
                trial.addFigure(fig, "/features/dataset%s_Kmeans/%s/%s-CM.png" % (trainDatasetName,folderPrefix, str(datasetNames)))
                plt.imshow(cm, interpolation='none', cmap="jet", aspect="auto", vmin=0, vmax=1.0)
                plt.ylabel("True Label")
                plt.xlabel("Kmeans Cluster Id")
                plt.colorbar()
                ax = plt.gca()

                for (j,i),label in np.ndenumerate(cm):
                    label = "%.2f" % label
                    ax.text(i,j,label,ha='center',va='center')
                    ax.text(i,j,label,ha='center',va='center')

            for i in range(len(allActivations)):
                print( "working on layer %d" % (i))
                kmeans = KMeans(n_clusters=n_clusters, random_state=0)
                kmeans.fit(trainActivations[i])
                # kmeans.cluster_centers_
                plotKmeansStuff(allActivations[i], kmeans, allLabels, markers, folderPrefix="/%s/layer%d" % (outputFolderPrefix,i))
                plotKmeansClassifications(allActivations[i], kmeans, allLabels, markers, folderPrefix="/%s/layer%d" % (outputFolderPrefix,i))
                # code.interact(local=dict(globals(), **locals()))

def plotKmeansStuff(trial, features, kmeans, labels, markers, folderPrefix="layer0"):
    colors = [key for key in mcolors.TABLEAU_COLORS]

    allDists = kmeans.transform(features)
    clusterIds = kmeans.predict(features)
    distToCluster =  allDists[[i for i in range(allDists.shape[0])],clusterIds]

    clusterDists = []
    plotLabels = []
    for i in range(kmeans.n_clusters):
        plotLabels.append(str(i))
        idxs = np.where(clusterIds == i)
        dist = distToCluster[idxs]
        clusterDists.append(dist)
    clusterDists.append(distToCluster)
    plotLabels.append(str("All Groups"))

    fig = plt.figure()
    plt.boxplot(clusterDists, meanline=True, labels=plotLabels)
    # trial.addFigure(fig, "/features/dataset%s_Kmeans/%s/%s-boxplot.png" % (str(trainDatasetNames),folderPrefix, str(datasetNames)))
    trial.addFigure(fig, "/features/%s/boxplot.png" % (folderPrefix))

    # code.interact(local=dict(globals(), **locals()))

def plotKmeansClassifications(trial, features, kmeans, labels, markers, folderPrefix="layer0"):
    labels = labels.numpy()
    clusterIds = kmeans.predict(features)
    clusterPredictLabel = copy.deepcopy(clusterIds)
    # get true labels
    allIdxs = [] # allIdxs[clusterNumber] = set of indexes that correspond to class
    clusterIdToLapel = []
    for i in range(kmeans.n_clusters):
        clusterIdxs = np.where(clusterIds == i)
        clusterLabels = labels[clusterIdxs]
        clusterLabel = np.bincount(clusterLabels).argmax()
        clusterPredictLabel[clusterIdxs] = clusterLabel
        clusterIdToLapel.append(clusterLabel)
        allIdxs.append(np.where(clusterIds == i))

    percentCorrect = float(np.sum(np.where(clusterPredictLabel == labels, 1, 0))) / float(labels.shape[0])

    cm = metrics.confusion_matrix(labels, clusterIds, normalize="true")
    fig = plt.figure()
    # trial.addFigure(fig, "/features/dataset%s_Kmeans/%s/%s-CM.png" % (str(trainDatasetNames),folderPrefix, str(datasetNames)))
    trial.addFigure(fig, "/features/%s/CM-clusterIds-normalizeTrue.png" % (folderPrefix))
    plt.imshow(cm, interpolation='none', cmap="jet", aspect="auto", vmin=0, vmax=1.0)
    plt.ylabel("True Label")
    plt.xlabel("Kmeans Cluster Id")
    plt.colorbar()
    plt.title("Cluster Ids | Normalize True")
    ax = plt.gca()
    for (j,i),label in np.ndenumerate(cm):
        label = "%.2f" % label
        ax.text(i,j,label,ha='center',va='center')
        ax.text(i,j,label,ha='center',va='center')

    cm = metrics.confusion_matrix(labels, clusterIds, normalize="pred")
    fig = plt.figure()
    # trial.addFigure(fig, "/features/dataset%s_Kmeans/%s/%s-CM.png" % (str(trainDatasetNames),folderPrefix, str(datasetNames)))
    trial.addFigure(fig, "/features/%s/CM-clusterIds-normalizePred.png" % (folderPrefix))
    plt.imshow(cm, interpolation='none', cmap="jet", aspect="auto", vmin=0, vmax=1.0)
    plt.ylabel("True Label")
    plt.xlabel("Kmeans Cluster Id")
    plt.colorbar()
    plt.title("ClusterIds | Normalize pred")
    ax = plt.gca()
    for (j,i),label in np.ndenumerate(cm):
        label = "%.2f" % label
        ax.text(i,j,label,ha='center',va='center')
        ax.text(i,j,label,ha='center',va='center')

    cm = metrics.confusion_matrix(labels, clusterPredictLabel, normalize="true")
    fig = plt.figure()
    trial.addFigure(fig, "/features/%s/CM-trueclusterIds-normalizeTrue.png" % (folderPrefix))
    plt.imshow(cm, interpolation='none', cmap="jet", aspect="auto", vmin=0, vmax=1.0)
    plt.ylabel("True Label")
    plt.xlabel("Kmeans Cluster Id")
    plt.colorbar()
    plt.title("True Cluster Ids | Normalize True | Percent Correct %f" % percentCorrect)
    ax = plt.gca()
    for (j,i),label in np.ndenumerate(cm):
        label = "%.2f" % label
        ax.text(i,j,label,ha='center',va='center')
        ax.text(i,j,label,ha='center',va='center')

    cm = metrics.confusion_matrix(labels, clusterPredictLabel, normalize="pred")
    fig = plt.figure()
    # trial.addFigure(fig, "/features/dataset%s_Kmeans/%s/%s-CM.png" % (str(trainDatasetNames),folderPrefix, str(datasetNames)))
    trial.addFigure(fig, "/features/%s/CM-trueclusterIds-normalizePred.png" % (folderPrefix))
    plt.imshow(cm, interpolation='none', cmap="jet", aspect="auto", vmin=0, vmax=1.0)
    plt.ylabel("True Label")
    plt.xlabel("Kmeans Cluster Id")
    plt.colorbar()
    plt.title("True ClusterIds | Normalize pred | Percent Correct %f" % percentCorrect)
    ax = plt.gca()
    for (j,i),label in np.ndenumerate(cm):
        label = "%.2f" % label
        ax.text(i,j,label,ha='center',va='center')
        ax.text(i,j,label,ha='center',va='center')

def getKMeansDiff(
    data
    , modelName="model"
    , modelNameloadModelPath="../simulations/multiDistortionTest/trial0/stateDict/modelStateDict0.pth"
    , trainDatasetNames=["task1ValidData"]
    , datasetNames=["task1ValidData-Blur-1", "task1ValidData-Blur-2", "task1ValidData-Blur-3"]
    , datasetSymbols=[".", "x", "x", "x"]
    , outputFolderPrefix="PreSleep"
    , n_clusters=10
    ):


    for sim in data.sims:
        for trial in sim.trials: 
            simObj = Simulation(trial.config, **trial.config)
            simObj.createSimMembers(trial.config["members"], trial.config["modifiers"], trial.config["stages"], cachedSimMembers={})
            model = simObj.members[modelName]
            model.load_state_dict(torch.load(modelNameloadModelPath, map_location=devices.comp))
            datasets = [simObj.members[d] for d in datasetNames]
            datasets = [getAllDataFromDataset(d) for d in datasets]

            trainDatasets = [simObj.members[d] for d in trainDatasetNames]
            trainDatasets = [getAllDataFromDataset(d) for d in trainDatasets]

            trainActivations , trainLabels , trainMarkers  = getActivationsForDatasets(model, trainDatasets, datasetSymbols) 
            allActivations, allLabels, markers  = getActivationsForDatasets(model, datasets, datasetSymbols)


            for i in range(len(allActivations)):
                print( "working on layer %d" % (i))
                kmeans = KMeans(n_clusters=n_clusters, random_state=0)
                kmeans.fit(trainActivations[i])
                # kmeans.cluster_centers_
                plotKmeansStuff(trial, allActivations[i], kmeans, allLabels, markers, folderPrefix="/kmeansDiff/%s/training-%s/testing-%s/layer%d" % (outputFolderPrefix,str(trainDatasetNames), str(datasetNames), i))
                plotKmeansClassifications(trial, allActivations[i], kmeans, allLabels, markers, folderPrefix="/kmeansDiff/%s/training-%s/testing-%s/layer%d" % (outputFolderPrefix,str(trainDatasetNames), str(datasetNames), i))
                # code.interact(local=dict(globals(), **locals()))

# def getConvLayerGradientMetric(
    # data
    # , modelName="model"
    # , modelNameloadModelPaths=["/stateDict/modelStateDict0.pth"]
    # , linePrettyNames = ["Post training"]
    # , datasetNames=["task1ValidData", "task1ValidData-Blur-1", "task1ValidData-Blur-2", "task1ValidData-Blur-3"]
    # , datsetValues=[0, 1, 2, 3]
    # , prettyXTicks=True
    # ):


    # for sim in data.sims:
        # for trial in sim.trials: 
            # plts = [plt.subplots(1, 2) for i in range(5)]
            # cmap = plt.get_cmap("tab10")
            # for pathIdx,modelNameloadModelPath in enumerate(modelNameloadModelPaths):
                # simObj = Simulation(trial.config, **trial.config)
                # simObj.createSimMembers(trial.config["members"], trial.config["modifiers"], trial.config["stages"], cachedSimMembers={})
                # model = simObj.members[modelName]
                # modelNameloadModelPathFull = trial.path + modelNameloadModelPath

                # model.load_state_dict(torch.load(modelNameloadModelPathFull, map_location=devices.comp))
                # datasets = [simObj.members[d] for d in datasetNames]
                # datasets = [getAllDataFromDataset(d) for d in datasets]

                # xs = []
                # Means = None
                # Stds = None
                # prettyXTicks = []
                # for dIdx, dataset in enumerate(datasets):
                    # prettyXTicks.append("%s %s" % (datasetNames[dIdx], str(datsetValues[dIdx])))
                    # allActivations, allLabels, markers  = getActivationsForDatasets(model, [dataset], datasetSymbols=None, flatten=False)
                    # if Means is None and Stds is None:
                        # # Means = [[0.0] * len(datasetNames)] * len(allActivations) # every layer has its own mean and std
                        # # Stds = [[0.0] * len(datasetNames)] * len(allActivations) # every layer has its own mean and std
                        # Means = [[0.0] * len(datasetNames) for ii in range(len(allActivations))] # every layer has its own mean and std
                        # Stds = [[0.0] * len(datasetNames) for ii in range(len(allActivations))] # every layer has its own mean and std
                    # xs.append(datsetValues[dIdx])
                    # for lIdx,layerActivations in enumerate(allActivations):
                        # if len(layerActivations.shape) <= 2: # if feed forward then skip
                            # continue
                        # channels = layerActivations.size()[1]
                        # sobelV = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32) / 4.0
                        # sobelKernelV = sobelV.unsqueeze(0).expand(1, 1, 3, 3)
                        # sobelH = sobelV.clone().T
                        # sobelKernelH = sobelH.unsqueeze(0).expand(1, 1, 3, 3)
                        # channelValues = torch.tensor([])
                        # for c in range(channels):
                            # try:
                                # channelData = layerActivations[:, c, :, :].unsqueeze(1)
                            # except:
                                # code.interact(local=dict(globals(), **locals()))
                            # gradV = torch.nn.functional.conv2d(channelData, sobelKernelV, bias=None, stride=1, padding=0, dilation=1, groups=1)
                            # gradH = torch.nn.functional.conv2d(channelData, sobelKernelH, bias=None, stride=1, padding=0, dilation=1, groups=1)

                            # ssum = (gradV.pow(2) + gradH.pow(2)).pow(0.5)
                            # ssum = ssum[torch.where(ssum != 0)] # grab nonnegative gradient values
                            # channelValues = torch.cat([channelValues, ssum], dim=0) # flattens all features to single dimmension

                        # Means[lIdx][dIdx] = channelValues.mean().item()
                        # # Stds[lIdx][dIdx] = channelValues.std().item()
                        # Stds[lIdx][dIdx] = channelValues.std().item() / Means[lIdx][dIdx]

                # for i in range(len(Means)): # i is layer index
                    # Means[i] = np.array(Means[i])
                    # Stds[i] = np.array(Stds[i])

                    # plts[i][1][0].plot(xs, Means[i], color=cmap(pathIdx))
                    # plts[i][1][0].set_title("Means")
                    # plts[i][1][1].plot(xs, Stds[i], color=cmap(pathIdx))
                    # plts[i][1][1].set_title("Normalized Stds")

                    # # fig= plt.figure()
                    # # plt.plot(xs, Means[i], color="tab:orange")
                    # # plt.fill_between(xs, Means[i]+Stds[i], Means[i]-Stds[i], color="tab:orange", alpha=0.4)
                    # # if prettyXTicks:
                        # # plt.xticks(xs, prettyXTicks, rotation = 90)
                    # # plt.tight_layout()
                    # # trial.addFigure(fig, "/features/gradientMetric/%s/layer%d/gradientMEtricOverDataset.png" % (str(datasetNames), i))
            # for i in  range(len(plts)):
                # if prettyXTicks:
                    # plts[i][1][0].set_xticks(xs, prettyXTicks, rotation = 90)
                    # plts[i][1][1].set_xticks(xs, prettyXTicks, rotation = 90)
                # plts[i][1][1].legend(linePrettyNames, loc='center left', bbox_to_anchor=(1, 0.5))
                # plts[i][0].tight_layout()
                # trial.addFigure(plts[i][0], "/features/gradientMetric/%s/layer%d/gradientMEtricOverDataset.png" % (str(datasetNames), i))

def getConvLayerGradientMetric(
    datas
    , modelName="model"
    , modelNameloadModelPaths=["/stateDict/modelStateDict0.pth"]
    , linePrettyNames = ["Post training"]
    , datasetNames=["task1ValidData", "task1ValidData-Blur-1", "task1ValidData-Blur-2", "task1ValidData-Blur-3"]
    , datsetValues=[0, 1, 2, 3]
    , prettyXTicks=True
    ):

    numLayers = 5
    plts = [plt.subplots(1, 2) for i in range(numLayers)] 
    cmap = plt.get_cmap("tab10")
    for ii, data in enumerate(datas):
        for sim in data.sims:
            for trial in sim.trials: 
                modelNameloadModelPath = modelNameloadModelPaths[ii]
                simObj = Simulation(trial.config, **trial.config)
                simObj.createSimMembers(trial.config["members"], trial.config["modifiers"], trial.config["stages"], cachedSimMembers={})
                model = simObj.members[modelName]
                modelNameloadModelPathFull = trial.path + modelNameloadModelPath

                model.load_state_dict(torch.load(modelNameloadModelPathFull, map_location=devices.comp))
                datasets = [simObj.members[d] for d in datasetNames]
                datasets = [getAllDataFromDataset(d) for d in datasets]

                xs = []
                Means = None
                Stds = None
                prettyXTicks = []
                for dIdx, dataset in enumerate(datasets):
                    prettyXTicks.append("%s %s" % (datasetNames[dIdx], str(datsetValues[dIdx])))
                    allActivations, allLabels, markers  = getActivationsForDatasets(model, [dataset], datasetSymbols=None, flatten=False)
                    if Means is None and Stds is None:
                        # Means = [[0.0] * len(datasetNames)] * len(allActivations) # every layer has its own mean and std
                        # Stds = [[0.0] * len(datasetNames)] * len(allActivations) # every layer has its own mean and std
                        Means = [[0.0] * len(datasetNames) for ii in range(len(allActivations))] # every layer has its own mean and std
                        Stds = [[0.0] * len(datasetNames) for ii in range(len(allActivations))] # every layer has its own mean and std
                    xs.append(datsetValues[dIdx])
                    for lIdx,layerActivations in enumerate(allActivations):
                        if len(layerActivations.shape) <= 2: # if feed forward then skip
                            continue
                        channels = layerActivations.size()[1]
                        sobelV = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32) / 4.0
                        sobelKernelV = sobelV.unsqueeze(0).expand(1, 1, 3, 3)
                        sobelH = sobelV.clone().T
                        sobelKernelH = sobelH.unsqueeze(0).expand(1, 1, 3, 3)
                        channelValues = torch.tensor([])
                        for c in range(channels):
                            try:
                                channelData = layerActivations[:, c, :, :].unsqueeze(1)
                            except:
                                code.interact(local=dict(globals(), **locals()))
                            gradV = torch.nn.functional.conv2d(channelData, sobelKernelV, bias=None, stride=1, padding=0, dilation=1, groups=1)
                            gradH = torch.nn.functional.conv2d(channelData, sobelKernelH, bias=None, stride=1, padding=0, dilation=1, groups=1)

                            ssum = (gradV.pow(2) + gradH.pow(2)).pow(0.5)
                            ssum = ssum[torch.where(ssum != 0)] # grab nonnegative gradient values
                            channelValues = torch.cat([channelValues, ssum], dim=0) # flattens all features to single dimmension

                        Means[lIdx][dIdx] = channelValues.mean().item()
                        # Stds[lIdx][dIdx] = channelValues.std().item()
                        Stds[lIdx][dIdx] = channelValues.std().item() / Means[lIdx][dIdx]

                for i in range(len(Means)): # i is layer index
                    Means[i] = np.array(Means[i])
                    Stds[i] = np.array(Stds[i])

                    plts[i][1][0].plot(xs, Means[i], color=cmap(ii))
                    plts[i][1][0].set_title("Means")
                    plts[i][1][1].plot(xs, Stds[i], color=cmap(ii))
                    plts[i][1][1].set_title("Normalized Stds")

                    # fig= plt.figure()
                    # plt.plot(xs, Means[i], color="tab:orange")
                    # plt.fill_between(xs, Means[i]+Stds[i], Means[i]-Stds[i], color="tab:orange", alpha=0.4)
                    # if prettyXTicks:
                        # plt.xticks(xs, prettyXTicks, rotation = 90)
                    # plt.tight_layout()
                    # trial.addFigure(fig, "/features/gradientMetric/%s/layer%d/gradientMEtricOverDataset.png" % (str(datasetNames), i))
            for i in  range(len(plts)):
                if prettyXTicks:
                    plts[i][1][0].set_xticks(xs, prettyXTicks, rotation = 90)
                    plts[i][1][1].set_xticks(xs, prettyXTicks, rotation = 90)
                plts[i][1][1].legend(linePrettyNames, loc='center left', bbox_to_anchor=(1, 0.5))
                plts[i][0].tight_layout()
                data.addFigure(plts[i][0], "/features/gradientMetric/%s/layer%d/gradientMEtricOverDataset.png" % (str(datasetNames), i))

def getConvLayerMagnitude(
    data
    , modelName="model"
    , modelNameloadModelPaths=["/stateDict/modelStateDict0.pth"]
    , linePrettyNames = ["Post training"]
    , datasetNames=["task1ValidData", "task1ValidData-Blur-1", "task1ValidData-Blur-2", "task1ValidData-Blur-3"]
    , datsetValues=[0, 1, 2, 3]
    , prettyXTicks=True
    ):


    for sim in data.sims:
        for trial in sim.trials: 
            plts = [plt.subplots(1, 3) for i in range(5)]
            cmap = plt.get_cmap("tab10")
            for pathIdx,modelNameloadModelPath in enumerate(modelNameloadModelPaths):
                simObj = Simulation(trial.config, **trial.config)
                simObj.createSimMembers(trial.config["members"], trial.config["modifiers"], trial.config["stages"], cachedSimMembers={})
                model = simObj.members[modelName]
                modelNameloadModelPathFull = trial.path + modelNameloadModelPath

                model.load_state_dict(torch.load(modelNameloadModelPathFull, map_location=devices.comp))
                datasets = [simObj.members[d] for d in datasetNames]
                datasets = [getAllDataFromDataset(d) for d in datasets]

                xs = []
                Means = None
                Stds = None
                NumZeros = None
                prettyXTicks = []
                for dIdx, dataset in enumerate(datasets):
                    prettyXTicks.append("%s %s" % (datasetNames[dIdx], str(datsetValues[dIdx])))
                    allActivations, allLabels, markers  = getActivationsForDatasets(model, [dataset], datasetSymbols=None, flatten=False)
                    if Means is None and Stds is None:
                        # Means = [[0.0] * len(datasetNames)] * len(allActivations) # every layer has its own mean and std
                        # Stds = [[0.0] * len(datasetNames)] * len(allActivations) # every layer has its own mean and std
                        Means = [[0.0] * len(datasetNames) for ii in range(len(allActivations))] # every layer has its own mean and std
                        Stds = [[0.0] * len(datasetNames) for ii in range(len(allActivations))] # every layer has its own mean and std
                        NumZeros = [[0.0] * len(datasetNames) for ii in range(len(allActivations))] # every layer has its own mean and std
                    xs.append(datsetValues[dIdx])
                    for lIdx,layerActivations in enumerate(allActivations):
                        if len(layerActivations.shape) <= 2: # if feed forward then skip
                            continue
                        channels = layerActivations.size()[1]
                        # sobelV = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32) / 4.0
                        # sobelKernelV = sobelV.unsqueeze(0).expand(1, 1, 3, 3)
                        # sobelH = sobelV.clone().T
                        # sobelKernelH = sobelH.unsqueeze(0).expand(1, 1, 3, 3)
                        channelValues = torch.tensor([])
                        for c in range(channels):
                            try:
                                channelData = layerActivations[:, c, :, :].unsqueeze(1)
                            except:
                                code.interact(local=dict(globals(), **locals()))
                            # gradV = torch.nn.functional.conv2d(channelData, sobelKernelV, bias=None, stride=1, padding=0, dilation=1, groups=1)
                            # gradH = torch.nn.functional.conv2d(channelData, sobelKernelH, bias=None, stride=1, padding=0, dilation=1, groups=1)

                            # ssum = (gradV.pow(2) + gradH.pow(2)).pow(0.5)
                            # channelValues = torch.cat([channelValues, ssum], dim=0) 
                            channelValues = torch.cat([channelValues, channelData], dim=0) 

                        Means[lIdx][dIdx] = channelValues.mean().item()
                        Stds[lIdx][dIdx] = channelValues.std().item()
                        totalActivations = np.prod([ss for ss in channelValues.shape])
                        NumZeros[lIdx][dIdx] = torch.sum(torch.where(channelValues == 0.0, torch.tensor(1.0),torch.tensor(0.0)) ).item() / float(totalActivations)

                for i in range(len(Means)): # i is layer index
                    Means[i] = np.array(Means[i])
                    Stds[i] = np.array(Stds[i])

                    plts[i][1][0].plot(xs, Means[i], color=cmap(pathIdx))
                    plts[i][1][0].set_title("Means")
                    plts[i][1][1].plot(xs, Stds[i], color=cmap(pathIdx))
                    plts[i][1][1].set_title("Stds")
                    plts[i][1][2].plot(xs, NumZeros[i], color=cmap(pathIdx))
                    plts[i][1][2].set_title("Numer of Zeros")

                    # fig= plt.figure()
                    # plt.plot(xs, Means[i], color="tab:orange")
                    # plt.fill_between(xs, Means[i]+Stds[i], Means[i]-Stds[i], color="tab:orange", alpha=0.4)
                    # if prettyXTicks:
                        # plt.xticks(xs, prettyXTicks, rotation = 90)
                    # plt.tight_layout()
                    # trial.addFigure(fig, "/features/gradientMetric/%s/layer%d/gradientMEtricOverDataset.png" % (str(datasetNames), i))
            for i in  range(len(plts)):
                if prettyXTicks:
                    plts[i][1][0].set_xticks(xs, prettyXTicks, rotation = 90)
                    plts[i][1][1].set_xticks(xs, prettyXTicks, rotation = 90)
                    plts[i][1][2].set_xticks(xs, prettyXTicks, rotation = 90)
                plts[i][1][2].legend(linePrettyNames, loc='center left', bbox_to_anchor=(1, 0.5))
                plts[i][0].tight_layout()
                trial.addFigure(plts[i][0], "/features/gradientMetric/%s/layer%d/magnitudeMetricOverDataset.png" % (str(datasetNames), i))

def getConvLayerGradientDist(
    data
    , modelName="model"
    , modelNameloadModelPaths=["/stateDict/modelStateDict0.pth"]
    , linePrettyNames = ["Post training"]
    , datasetNames=["task1ValidData", "task1ValidData-Blur-1", "task1ValidData-Blur-2", "task1ValidData-Blur-3"]
    , datsetValues=[0, 1, 2, 3]
    , prettyXTicks=True
    ):


    for sim in data.sims:
        for trial in sim.trials: 
            plts = [plt.subplots(1, 2) for i in range(5)]
            cmap = plt.get_cmap("tab10")
            for pathIdx,modelNameloadModelPath in enumerate(modelNameloadModelPaths):
                simObj = Simulation(trial.config, **trial.config)
                simObj.createSimMembers(trial.config["members"], trial.config["modifiers"], trial.config["stages"], cachedSimMembers={})
                model = simObj.members[modelName]
                modelNameloadModelPathFull = trial.path + modelNameloadModelPath

                model.load_state_dict(torch.load(modelNameloadModelPathFull, map_location=devices.comp))
                datasets = [simObj.members[d] for d in datasetNames]
                datasets = [getAllDataFromDataset(d) for d in datasets]

                xs = []
                gradVals = None
                Stds = None
                prettyXTicks = []
                for dIdx, dataset in enumerate(datasets):
                    prettyXTicks.append("%s %s" % (datasetNames[dIdx], str(datsetValues[dIdx])))
                    allActivations, allLabels, markers  = getActivationsForDatasets(model, [dataset], datasetSymbols=None, flatten=False)
                    if gradVals is None and Stds is None:
                        # gradVals = [[0.0] * len(datasetNames)] * len(allActivations) # every layer has its own mean and std
                        # Stds = [[0.0] * len(datasetNames)] * len(allActivations) # every layer has its own mean and std
                        gradVals = [[0.0] * len(datasetNames) for ii in range(len(allActivations))] # every layer has its own mean and std
                        Stds = [[0.0] * len(datasetNames) for ii in range(len(allActivations))] # every layer has its own mean and std
                    xs.append(datsetValues[dIdx])
                    for lIdx,layerActivations in enumerate(allActivations):
                        if len(layerActivations.shape) <= 2: # if feed forward then skip
                            continue
                        channels = layerActivations.size()[1]
                        sobelV = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32) / 4.0
                        sobelKernelV = sobelV.unsqueeze(0).expand(1, 1, 3, 3)
                        sobelH = sobelV.clone().T
                        sobelKernelH = sobelH.unsqueeze(0).expand(1, 1, 3, 3)
                        channelValues = torch.tensor([])
                        for c in range(channels):
                            try:
                                channelData = layerActivations[:, c, :, :].unsqueeze(1)
                            except:
                                code.interact(local=dict(globals(), **locals()))
                            gradV = torch.nn.functional.conv2d(channelData, sobelKernelV, bias=None, stride=1, padding=0, dilation=1, groups=1)
                            gradH = torch.nn.functional.conv2d(channelData, sobelKernelH, bias=None, stride=1, padding=0, dilation=1, groups=1)

                            ssum = (gradV.pow(2) + gradH.pow(2)).pow(0.5)
                            # ssum = ssum[torch.where(ssum != 0)] # grab nonnegative gradient values
                            channelValues = torch.cat([channelValues, ssum], dim=0) # flattens all features to single dimmension

                        gradVals[lIdx][dIdx] = channelValues.ravel().detach().numpy()
                        # Stds[lIdx][dIdx] = channelValues.std().item()

                for i in range(len(gradVals)): # i is layer index
                    gradVals[i] = np.array(gradVals[i])
                    # Stds[i] = np.array(Stds[i])

                    plts[i][1][0].hist(gradVals[i], alpha=0.5)
                    # plts[i][1][0].plot(xs, gradVals[i], color=cmap(pathIdx))
                    plts[i][1][0].set_title("gradVals")
                    # plts[i][1][1].plot(xs, Stds[i], color=cmap(pathIdx))
                    # plts[i][1][1].set_title("Stds")

                    # fig= plt.figure()
                    # plt.plot(xs, gradVals[i], color="tab:orange")
                    # plt.fill_between(xs, gradVals[i]+Stds[i], gradVals[i]-Stds[i], color="tab:orange", alpha=0.4)
                    # if prettyXTicks:
                        # plt.xticks(xs, prettyXTicks, rotation = 90)
                    # plt.tight_layout()
                    # trial.addFigure(fig, "/features/gradientMetric/%s/layer%d/gradientMEtricOverDataset.png" % (str(datasetNames), i))
            for i in  range(len(plts)):
                # if prettyXTicks:
                    # plts[i][1][0].set_xticks(xs, prettyXTicks, rotation = 90)
                    # plts[i][1][1].set_xticks(xs, prettyXTicks, rotation = 90)
                plts[i][1][0].legend(linePrettyNames, loc='center left', bbox_to_anchor=(1, 0.5))
                plts[i][0].tight_layout()
                trial.addFigure(plts[i][0], "/features/gradientMetric/%s/layer%d/gradientHistOverDataset.png" % (str(datasetNames), i))
