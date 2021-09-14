import matplotlib.pyplot as plt
import os

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
            plt.plot(trial.data.datasetMetrics[datsetName][metricName])
            if trial.data.datasetMetrics[datsetName][metricName][0] < trial.data.datasetMetrics[datsetName][metricName][1]:
                    # print(trial.path)
                    numImproved += 1
            if trial.data.datasetMetrics[datsetName][metricName][0] > trial.data.datasetMetrics[datsetName][metricName][1]:
                    # print(trial.path)
                    numDecreased += 1

    print("%s %s | %d increased | %d decreased | %d total number of trials" % (datsetName, metricName, numImproved, numDecreased, totalNumTrials))

# plots every combination of  datasetName and metricName
def plotTrialMetrics(data, datsetNames=["task1TrainData"], metricNames=["confidence"]):
    for sim in data.sims:
        for trial in sim.trials:
            leg = []
            fig = plt.figure()
            trial.addFigure(fig, "%s-%s.png" % (str(datsetNames), str(metricNames)))
            for datasetName in datsetNames:
                for metricName in metricNames:
                    plt.plot(trial.data.datasetMetrics[datasetName][metricName])
                    leg.append("%s %s" % (datasetName, metricName))
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
    