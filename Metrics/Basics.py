import matplotlib.pyplot as plt
import os
import numpy as np

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