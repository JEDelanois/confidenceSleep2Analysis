import matplotlib.pyplot as plt
import os
import numpy as np
import copy
import code
from tabulate import tabulate
import Utils.ConfigUtil



def plotAllTrialMetric(data, datsetName="task1TrainData", metricName="confidence"):
    print("Plotting %s" % metricName)
    fig = plt.figure()
    data.addFigure(fig, "%s-%s.pdf" % (datsetName, metricName))
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
                trial.addFigure(fig, "%s-%s-line.pdf" % (str(datsetNames), str(metricNames)))

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
                trial.addFigure(fig, "%s-%s-line.pdf" % (str(datsetNames), str(metricNames)))

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
                trial.addFigure(fig, "%s-%s-bar.pdf" % (str(datsetNames), str(metricNames)))

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
                trial.addFigure(fig, "%s-%s-metricOverDatasetValue.pdf" % (str(datsetNames), str(metricName)))

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

# x-axis - datset name gets mapped to dataset value and plotted 
# y-axis - metric value 
# every line corresponds the data set at a specific time point
def plotSpecificTrialMetricOverDatasetValue(datas, datsetNames=["task1TrainData"], datsetValues=[0], timePoints=[0], metricName="confidence", timePointsPrettyNames=None, prettyXTicks=True, prettyFileName=None, prettyXLabel=None, lineStyles=["-"], lineColors=["tab:blue"]):
    # code.interact(local=dict(globals(), **locals()))
    assert len(datas) == len(timePoints)


    leg = []
    fig = plt.figure()
    if prettyFileName is not None:
        for data in datas:
            data.addFigure(fig, prettyFileName)
    else:
        for data in datas:
            data.addFigure(fig, "%s-%s-specificMetricOverDatasetValue.pdf" % (str(datsetNames), str(metricName)))
    for t,(data, timePoint) in enumerate(zip(datas,timePoints)):
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

                plt.plot(xs,ys, lineStyles[t], c=lineColors[t] )
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

    


def meanPerformanceAtTimeGenerator(timePoint=0, datasetNames=[], metricName="matlabAcc"):
    def meanPerformanceAtTime(sim):
        ret = []
        for datasetName in datasetNames:
            ret.append(sim.trials[0].data.datasetMetrics[datasetName][metricName][timePoint, 1])
        return np.mean(ret)
    return meanPerformanceAtTime

# every sim gets the specifed value (x value)
# metricFunction gets the value (y value)
def plotMetricOverConfigValue(datas, configPath=["modifiers", 1, 1,"datasetPercentages", 0], simPerformanceFunction=lambda x:0, prettyFileName=None, datasNames=["Sim 1"], ylabel="matlabAcc", xlabel="Dataset Size", xscale="linear", title="Perofrmance"):
    fig = plt.figure()
    for data in datas:
        # code.interact(local=dict(globals(), **locals()))
        xValues = [Utils.ConfigUtil.getDictValueFromPath(sim.trials[0].config, configPath) for sim in data.sims]
        yValues = [simPerformanceFunction(sim) for sim in data.sims]
        data.addFigure(fig, prettyFileName)
        plt.plot(xValues, yValues)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    ax = plt.gca()
    ax.set_xscale(xscale)
    plt.tight_layout()
    plt.legend(datasNames)