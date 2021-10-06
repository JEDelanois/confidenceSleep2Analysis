import matplotlib.pyplot as plt
import os
import numpy as np

def plotAllTrialMetric(data, datsetName="task1TrainData", metricName="confidence"):
    print("Plotting %s" % metricName)
    fig = plt.figure()
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
def plotTrialMetrics(data, datsetNames=["task1TrainData"], metricNames=["confidence"]):
    for sim in data.sims:
        for trial in sim.trials:
            leg = []
            fig = plt.figure()
            trial.addFigure(fig, "%s-%s-line.png" % (str(datsetNames), str(metricNames)))
            for datasetName in datsetNames:
                for metricName in metricNames:
                    metric = trial.data.datasetMetrics[datasetName][metricName]
                    plt.plot(metric[:,0], metric[:,1])
                    leg.append("%s %s" % (datasetName, metricName))
            plt.legend(leg)

def barTrialMetrics(data, datsetNames=["task1TrainData"], metricNames=["confidence"], plotIdxs=[-1], xticks=None):
    numBarsPerStage = len(datsetNames)
    x = np.arange(len(plotIdxs))
    width = 1/(numBarsPerStage + 1)


    for sim in data.sims:
        for trial in sim.trials:
            fig = plt.figure()
            trial.addFigure(fig, "%s-%s-bar.png" % (str(datsetNames), str(metricNames)))
            shiftCounter = 0
            leg = []
            for datasetName in datsetNames:
                for metricName in metricNames:
                    leg.append("%s %s" % (datasetName, metricName))
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

    plt.legend(leg)

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
    