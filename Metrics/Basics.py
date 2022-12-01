import matplotlib.pyplot as plt
import os
import numpy as np
import copy
import code
from tabulate import tabulate
import Utils.ConfigUtil
import matplotlib.colors as mcolors



def plotAllTrialMetric(data, datsetName="task1TrainData", metricName="confidence"):
    print("Plotting %s" % metricName)
    fig = plt.figure()
    # data.addFigure(fig, "%s-%s.pdf" % (datsetName, metricName))
    data.addFigure(fig, "%s-%s.png" % (datsetName, metricName))
    totalNumTrials = 0
    numImproved = 0
    numDecreased = 0
    for sim in data.sims:
        for trial in sim.trials:
            totalNumTrials += 1
            metric = trial.data.datasetMetrics[datsetName][metricName]
            plt.plot(metric[:,0], metric[:,1])
            if metric[0, 1] < metric[-1, 1]:
                    # print(trial.path)
                    numImproved += 1
            if metric[0, 1] > metric[-1, 1]:
                    # print(trial.path)
                    numDecreased += 1

    print("%s %s | %d increased | %d decreased | %d total number of trials" % (datsetName, metricName, numImproved, numDecreased, totalNumTrials))

# plots every combination of  datasetName and metricName
def plotTrialMetrics(data, datsetNames=["task1TrainData"], metricNames=["confidence"], prettyFileName=None):
    for sim in data.sims:
        for trial in sim.trials:
            leg = []
            fig = plt.figure()

            if prettyFileName is not None:
                trial.addFigure(fig, prettyFileName)
            else:
                # trial.addFigure(fig, "%s-%s-line.pdf" % (str(datsetNames), str(metricNames)))
                trial.addFigure(fig, "%s-%s-line.png" % (str(datsetNames), str(metricNames)))

            for datasetName in datsetNames:
                for metricName in metricNames:
                    metric = trial.data.datasetMetrics[datasetName][metricName]
                    plt.plot(metric[:,0], metric[:,1])
                    leg.append("%s\n%s" % (datasetName, metricName))
            plt.legend(leg, loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout()

def plotTrialMetricsDiff(data, datsetNames=["task1TrainData"], metricNames=["confidence"], initialPoint=0, finalPoint=1, prettyFileName=None):
    for sim in data.sims:
        for trial in sim.trials:
            leg = []
            fig,axs = plt.subplots(nrows=2, ncols=1)

            if prettyFileName is not None:
                trial.addFigure(fig, prettyFileName)
            else:
                # trial.addFigure(fig, "%s-%s-line.pdf" % (str(datsetNames), str(metricNames)))
                trial.addFigure(fig, "%s-%s-line.png" % (str(datsetNames), str(metricNames)))

            axs[0].set_title("Initial %d - Final %d" % (initialPoint, finalPoint))

            allDiffs = []
            plotIdx = 0
            xticks = []
            for datasetName in datsetNames:
                for metricName in metricNames:
                    metric = trial.data.datasetMetrics[datasetName][metricName]
                    diffValue = metric[finalPoint, 1] - metric[initialPoint, 1]
                    allDiffs.append(diffValue)
                    xticks.append(plotIdx)
                    axs[0].scatter([plotIdx], [diffValue])
                    leg.append("%s\n%s" % (datasetName, metricName))
                    plotIdx += 1
            # axs[0].scatter(xticks, allDiffs)
            # axs[0].set_xticks(xticks)
            # axs[0].set_xticklabels(leg, rotation=45)
            mean = np.mean(allDiffs)
            axs[0].legend(leg, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='xx-small')
            axs[0].plot(xticks, np.repeat(mean, len(xticks)))
            axs[1].hist(allDiffs)
            plt.tight_layout()

def barTrialMetrics(data, datsetNames=["task1TrainData"], metricNames=["confidence"], plotIdxs=[-1], xticks=None, prettyFileName=None):
    numBarsPerStage = len(datsetNames)
    x = np.arange(len(plotIdxs))
    width = 1/(numBarsPerStage + 1)


    for sim in data.sims:
        for trial in sim.trials:
            fig = plt.figure()

            if prettyFileName is not None:
                trial.addFigure(fig, prettyFileName)
            else:
                # trial.addFigure(fig, "%s-%s-bar.pdf" % (str(datsetNames), str(metricNames)))
                trial.addFigure(fig, "%s-%s-bar.png" % (str(datsetNames), str(metricNames)))

            shiftCounter = 0
            leg = []
            for datasetName in datsetNames:
                for metricName in metricNames:
                    leg.append("%s\n%s" % (datasetName, metricName))
                    barsForStage = []
                    stageNames = []
                    for stageIdx in plotIdxs:
                        barVal = trial.data.datasetMetrics[datasetName][metricName][stageIdx, 1]
                        barsForStage.append(barVal)
                        stageNames.append(stageIdx)
                    shiftedX = x + (shiftCounter * width)
                    plt.bar(shiftedX, barsForStage, width=width)
                    shiftCounter += 1

            plt.xticks(x + ((numBarsPerStage - 1) * (width / 2)), stageNames)
            if xticks is not None:
                plt.xticks(x + ((numBarsPerStage - 1) * (width / 2)), xticks)

            plt.legend(leg, loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout()

# input - array to be sorted and printed out, can contain anything
# sotrKey - lambda function to sort array by
# printNum - how many of top of list to print
# fileName - file to print to
# folderPath - path to output directory
# SortReverse - if true sorted max to min, else sorted min to max
def printRankedMetric(input, sortReverse=False, sortKey=lambda x: x[0], printNum=10, folderPath="./figures/", fileName="out.txt"):
    input.sort(key=sortKey, reverse=sortReverse)
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)

    with open(os.path.join(folderPath, fileName), 'w') as outFile:
        if len(input) == 0: 
            outFile.write("No items in input\nIf sorting was performed then probably no items fit criteria")

        else:
            for i in range(printNum):
                outFile.write('%s\n' % (str(input[i])))
    
# x-axis - datset name gets mapped to dataset value and plotted 
# y-axis - metric value 
# every line corresponds to a different time point
def plotTrialMetricOverDatasetValue(data, datsetNames=["task1TrainData"], datsetValues=[0], timePoints=[0], metricName="confidence", timePointsPrettyNames=None, prettyXTicks=True, prettyFileName=None, prettyXLabel=None):
    for sim in data.sims:
        for trial in sim.trials:
            leg = []
            fig = plt.figure()

            if prettyFileName is not None:
                trial.addFigure(fig, prettyFileName)
            else:
                # trial.addFigure(fig, "%s-%s-metricOverDatasetValue.pdf" % (str(datsetNames), str(metricName)))
                trial.addFigure(fig, "%s-%s-metricOverDatasetValue.png" % (str(datsetNames), str(metricName)))

            for t,timePoint in enumerate(timePoints):
                xs = []
                ys = []
                prettyXTicks = []
                for i, datasetName in enumerate(datsetNames):
                    prettyXTicks.append("%s %s" % (datasetName, str(datsetValues[i])))
                    xs.append(datsetValues[i])
                    metricValue = trial.data.datasetMetrics[datasetName][metricName][timePoint,1]
                    ys.append(metricValue)

                plt.plot(xs,ys)
                if timePointsPrettyNames ==  None:
                    leg.append("TimePoint %s" % (str(timePoint)))
                else:
                    leg.append("TimePoint %s" % (timePointsPrettyNames[t]))

            plt.legend(leg, loc='center left', bbox_to_anchor=(1, 0.5))
            if prettyXTicks:
                plt.xticks(xs, prettyXTicks, rotation = 90)
            plt.ylabel(metricName)
            if prettyXLabel is not None:
                plt.xlabel(prettyXLabel)
            plt.tight_layout()

# # x-axis - datset name gets mapped to dataset value and plotted 
# # y-axis - metric value 
# # every line corresponds the data set at a specific time point
# def plotSpecificTrialMetricOverDatasetValue(datas, datsetNames=["task1TrainData"], datsetValues=[0], timePoints=[0], metricName="confidence", timePointsPrettyNames=None, prettyXTicks=True, prettyFileName=None, prettyXLabel=None, lineStyles=["-"], lineColors=["tab:blue"]):
    # # code.interact(local=dict(globals(), **locals()))
    # assert len(datas) == len(timePoints)


    # leg = []
    # fig = plt.figure()
    # if prettyFileName is not None:
        # for data in datas:
            # data.addFigure(fig, prettyFileName)
    # else:
        # for data in datas:
            # # data.addFigure(fig, "%s-%s-specificMetricOverDatasetValue.pdf" % (str(datsetNames), str(metricName)))
            # data.addFigure(fig, "%s-%s-specificMetricOverDatasetValue.png" % (str(datsetNames), str(metricName)))
    # for t,(data, timePoint) in enumerate(zip(datas,timePoints)):
        # for sim in data.sims:
            # for trial in sim.trials:


                # # for t,timePoint in enumerate(timePoints):
                # xs = []
                # ys = []
                # prettyXTicks = []
                # for i, datasetName in enumerate(datsetNames):
                    # prettyXTicks.append("%s %s" % (datasetName, str(datsetValues[i])))
                    # xs.append(datsetValues[i])
                    # metricValue = trial.data.datasetMetrics[datasetName][metricName][timePoint,1]
                    # ys.append(metricValue)

                # plt.plot(xs,ys, lineStyles[t], c=lineColors[t] )
                # if timePointsPrettyNames ==  None:
                    # leg.append("TimePoint %s" % (str(timePoint)))
                # else:
                    # leg.append("TimePoint %s" % (timePointsPrettyNames[t]))

        # plt.legend(leg, loc='center left', bbox_to_anchor=(1, 0.5))
        # if prettyXTicks:
            # plt.xticks(xs, prettyXTicks, rotation = 90)
        # plt.ylabel(metricName)
        # if prettyXLabel is not None:
            # plt.xlabel(prettyXLabel)
        # plt.tight_layout()

# x-axis - datset name gets mapped to dataset value and plotted 
# y-axis - metric value 
# every line corresponds the data set at a specific time point
def plotSpecificTrialMetricOverDatasetValue(datas, datsetNames=["task1TrainData"], datsetValues=[0], timePoints=[0], metricName="confidence", timePointsPrettyNames=None, usePrettyXTicks=True, prettyFileName=None, prettyXLabel=None, lineStyles=["-"], lineColors=["tab:blue"], alpha=0.3):
    # code.interact(local=dict(globals(), **locals()))
    assert len(datas) == len(timePoints)


    leg = []
    fig = plt.figure()
    if prettyFileName is not None:
        for data in datas:
            data.addFigure(fig, prettyFileName)
    else:
        for data in datas:
            data.addFigure(fig, "%s-%s-specificMetricOverDatasetValue.png" % (str(datsetNames), str(metricName)))
            data.addFigure(fig, "%s-%s-specificMetricOverDatasetValue.pdf" % (str(datsetNames), str(metricName)))
    if lineColors is None:
        lineColors = [k for k in mcolors.TABLEAU_COLORS.keys()]
    for t,(data, timePoint) in enumerate(zip(datas,timePoints)):
        allXs = []
        allYs = []
        for sim in data.sims:
            for trial in sim.trials:


                # for t,timePoint in enumerate(timePoints):
                xs = []
                ys = []
                prettyXTicks = []
                for i, datasetName in enumerate(datsetNames):
                    prettyXTicks.append("%s %s" % (datasetName, str(datsetValues[i])))
                    xs.append(datsetValues[i])
                    metricValue = trial.data.datasetMetrics[datasetName][metricName][timePoint,1]
                    ys.append(metricValue)
                allXs.append(xs)
                allYs.append(ys)
            meanXs = allXs[0] # all x values for datasets should be the same
            allYs = np.array(allYs)
            meanYs = allYs.mean(axis=0)
            stdsYs = allYs.std(axis=0)

            label = None
            if timePointsPrettyNames ==  None:
                label = "TimePoint %s" % (str(timePoint))
            else:
                label = "TimePoint %s" % (timePointsPrettyNames[t])
            plt.plot(meanXs,meanYs, lineStyles[t], c=lineColors[t], label=label)
            plt.fill_between(meanXs, meanYs+stdsYs, meanYs-stdsYs, color=lineColors[t], alpha=alpha)

    # plt.legend()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if usePrettyXTicks:
        plt.xticks(xs, prettyXTicks, rotation = 90)
    plt.ylabel(metricName)
    if prettyXLabel is not None:
        plt.xlabel(prettyXLabel)
    # plt.gca().set_ylim(top=1.0)
    plt.tight_layout()

# x-axis - dataset name
# y-axis - certain time point for certain data object
# every line corresponds the data set at a specific time point
def plotMetricTable(datas, modelNames=[], datsetNames=["task1TrainData"], timePoints=[0], metricName="confidence", timePointsPrettyNames=None, prettyXTicks=True, prettyFileName=None, prettyXLabel=None):
    # code.interact(local=dict(globals(), **locals()))
    assert len(datas) == len(timePoints)
    # code.interact(local=dict(globals(), **locals()))


    #________ imshow
    fig = plt.figure()
    if prettyFileName is not None:
        for data in datas:
            data.addFigure(fig, "%s-%s-imshow.png" % (prettyFileName, str(metricName)))
    else:
        for data in datas:
            data.addFigure(fig, "modelNames%s-DatasetNames%s-%s-imshow.png" % (str(datsetNames), str(datsetNames), str(metricName)))

    yticks = []
    table = []
    # right now this only works for a single trial
    for t,(data, timePoint) in enumerate(zip(datas,timePoints)):
        for sim in data.sims:
            for trial in sim.trials:
                # for t,timePoint in enumerate(timePoints):
                table.append([])
                for i, datasetName in enumerate(datsetNames):
                    metricValue = trial.data.datasetMetrics[datasetName][metricName][timePoint,1]
                    table[-1].append(metricValue)
                
                if timePointsPrettyNames ==  None:
                    yticks.append("%s TimePoint %s" % (str(timePoint)))
                else:
                    yticks.append("%s" % (timePointsPrettyNames[t]))
                # code.interact(local=dict(globals(), **locals()))

    

    xticks = copy.deepcopy(datsetNames) + ["Average"]
    table = np.array(table)
    table = np.hstack((table, np.expand_dims(table.mean(1), axis=1)))
    # vmin = 0.0
    # vmax = 1.0
    vmin = None
    vmax = None
    plt.imshow(table, interpolation='none', cmap="Reds", aspect="auto", vmin=vmin, vmax=vmax)
    ax = plt.gca()
    for (j,i),label in np.ndenumerate(table):
        label = "%.4f" % label
        ax.text(i,j,label,ha='center',va='center', size="small")
        ax.text(i,j,label,ha='center',va='center', size="small")

    plt.xticks([i for i in range(len(xticks))], xticks, rotation = 90)
    plt.yticks([i for i in range(len(yticks))], yticks)
    plt.title(metricName)
    plt.xlabel("Dataset")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.colorbar()


    #________ blot bar means
    fig = plt.figure()
    if prettyFileName is not None:
        for data in datas:
            data.addFigure(fig, "%s-%s-barMeans.png" % (prettyFileName, str(metricName)))
    else:
        for data in datas:
            data.addFigure(fig, "%s-%s-barMeans.png" % (str(datsetNames), str(metricName)))

    means = table[:,-1]
    mmin = means.min() - (0.05 * means.min())
    mmax = means.max() + (0.05 * means.max())
    plt.bar(np.arange(means.shape[0]), means)
    plt.xticks(np.arange(means.shape[0]), yticks, rotation=90)
    plt.ylim((mmin, mmax))
    plt.title("Mean %s" % (metricName))
    plt.ylabel(metricName)
    plt.tight_layout()





    #________ table

    fig, ax = plt.subplots(1, 1)
    if prettyFileName is not None:
        for data in datas:
            data.addFigure(fig, "%s-%s-table.png" % (prettyFileName, str(metricName)))
    else:
        for data in datas:
            data.addFigure(fig, "%s-%s-table.png" % (str(datsetNames), str(metricName)))
    
    ax.axis('tight')
    ax.axis('off')
    the_table = ax.table(table, rowLabels=yticks, colLabels=xticks, cellLoc="center")
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    ax.set_title(metricName)
    plt.tight_layout()

    fileOutputPaths = []
    if prettyFileName is not None:
        for data in datas:
            fileOutputPaths.append("%s/figures/%s-%s-table.txt" % (data.figureFolderPath, prettyFileName, str(metricName)))
    else:
        for data in datas:
            fileOutputPaths.append("%s/figures/%s-%s-table.txt" % (data.figureFolderPath, str(datsetNames), str(metricName)))

    #________ tabulate table

    listTable = table.tolist()
    for i,row in enumerate(listTable):
        row.insert(0, yticks[i])

    s = tabulate(listTable, headers=xticks)
    for filePath in fileOutputPaths:
        os.makedirs(os.path.dirname(filePath), exist_ok=True)
        with open(filePath, "w") as f:
            f.write(s)


def plotMultiMetricTable(
    data
    , datasetForm="testTask1AllData-Blur-%s-SP-%s"

    , dataset0Vals=[1, 3, 6]
    , dataset1Vals=[0.1, 0.3, 0.6]

    ,dataset0_0Fallback="testTask1AllData-SP-%s"
    ,dataset1_0Fallback="testTask1AllData-Blur-%s"
    ,both_0Fallback="testTask1AllData"

    , data0Label="Blur Intensity"
    , data1Label="SP Intensity"

    , timePoint=0
    , metricName="matlabAcc"
    , timePointsPrettyName=None
    , prettyFileName=None

    ,vmin=None
    ,vmax=None
    ):

    def prettyNum(num):
        return str(num).replace(".", "_")



    for sim in data.sims:
        fig = plt.figure()

        if prettyFileName is not None:
            sim.addFigure(fig, "%s-%s-MultiMetricImshow.png" % (prettyFileName, str(metricName)))
        else:
            sim.addFigure(fig, "%s-%s-MultiMetricImshow.png" % (datasetForm, metricName))

        plt.title("%s\non\n%s" % (sim.title, datasetForm))

        table = []
        for i in range(len(dataset0Vals)):
            table.append([])
            for j in range(len(dataset1Vals)):
                table[i].append(None) # now can use table[i][j]
                trialVals = []
                for trial in sim.trials:

                    # get correct dataset name base on falbacks
                    datasetName = None
                    if dataset0Vals[i] == 0 and dataset1Vals[j] == 0:
                        datasetName = both_0Fallback
                    elif dataset0Vals[i] == 0:
                        datasetName = dataset0_0Fallback % prettyNum(dataset1Vals[j])
                    elif dataset1Vals[j] == 0:
                        datasetName = dataset1_0Fallback % prettyNum(dataset0Vals[i])
                    else:
                        datasetName = datasetForm % (prettyNum(dataset0Vals[i]), prettyNum(dataset1Vals[j])) 

                    # error checking
                    if datasetName is None:
                        print("invalid dataset name")
                        print(datasetName)
                        exit()

                    metricValue = trial.data.datasetMetrics[datasetName][metricName][timePoint,1]
                    trialVals.append(metricValue)
                table[i][j] = np.mean(trialVals)


        table = np.array(table)
        plt.imshow(table, interpolation='none', cmap="Reds", aspect="auto", vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.xlabel(data1Label)
        plt.ylabel(data0Label)
        ax = plt.gca()
        for (j,i),label in np.ndenumerate(table):
            label = "%.4f" % label
            ax.text(i,j,label,ha='center',va='center', size="small")
            ax.text(i,j,label,ha='center',va='center', size="small")

        ax.set_xticks(np.arange(len(dataset1Vals)))
        ax.set_yticks(np.arange(len(dataset0Vals)))
        ax.set_xticklabels(dataset1Vals)
        ax.set_yticklabels(dataset0Vals)
        plt.tight_layout()
    

    


def meanPerformanceAtTimeGenerator(timePoint=0, datasetNames=[], metricName="matlabAcc"):

    def meanPerformanceAtTime(sim):
        ret = []
        for trial in sim.trials:
            trialPerf = []
            for datasetName in datasetNames:
                trialPerf.append(trial.data.datasetMetrics[datasetName][metricName][timePoint, 1])
            ret.append(np.mean(trialPerf))
        return np.mean(ret), np.std(ret)

    return meanPerformanceAtTime

# every sim gets the specifed value (x value)
# metricFunction gets the value (y value)
# config path will be used for xValues if xvalues is not specified
# xValues is the list of values where each entry is the x values for the coresponding simulations
def plotMetricOverConfigValue(datas, horizontalLineDatas=None , xValuess=None, configPath=["modifiers", 1, 1,"datasetPercentages", 0], simPerformanceFunction=lambda x:0, prettyFileName=None, datasNames=["Sim 1"], horizontalDatasNames=["Baseline"], lineData=None, horizontalLineData=None, ylabel="matlabAcc", xlabel="Dataset Size", xscale="linear", title="Perofrmance", alpha=0.1):
    fig = plt.figure()
    for i,data in enumerate(datas):
        # code.interact(local=dict(globals(), **locals()))
        if xValuess is None:
            xValues = [Utils.ConfigUtil.getDictValueFromPath(sim.trials[0].config, configPath) for sim in data.sims]
        else:
            xValues = xValuess[i]
        yValues = [simPerformanceFunction(sim) for sim in data.sims]
        # code.interact(local=dict(globals(), **locals()))
        yMeans = np.array([t[0] for t in yValues])
        yStds = np.array([t[1] for t in yValues])
        data.addFigure(fig, prettyFileName)
        if lineData is None:
            plt.plot(xValues, yMeans, label=datasNames[i])
        else:
            plt.plot(xValues, yMeans, lineData[i]["style"], c=lineData[i]["color"], label=datasNames[i])
            plt.fill_between(xValues, yMeans+yStds, yMeans-yStds, color=lineData[i]["color"], alpha=alpha)

    if horizontalLineDatas is not None:
        for i,data in enumerate(horizontalLineDatas):
            # code.interact(local=dict(globals(), **locals()))
            meanValue, stdValue = simPerformanceFunction(data.sims[0])
            # code.interact(local=dict(globals(), **locals()))
            yMeans = np.array([meanValue] * len(xValues))
            yStds = np.array([stdValue] * len(xValues))
            if horizontalLineData is None:
                plt.plot(xValues, yMeans, label=horizontalDatasNames[i])
            else:
                plt.plot(xValues, yMeans, horizontalLineData[i]["style"], c=horizontalLineData[i]["color"], label=horizontalDatasNames[i])
                plt.fill_between(xValues, yMeans+yStds, yMeans-yStds, color=horizontalLineData[i]["color"], alpha=alpha)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    ax = plt.gca()
    ax.set_xscale(xscale)
    # plt.legend()
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(loc='lower right', fontsize="small")
    plt.tight_layout()