
from os import wait
import Utils.DataUtil

def loadClassMetric2(data):
    print("Loading ClassMetric2")
    loadMetric(data, metricName="classMetric2", metricFile="classMetric2.txt")

def loadConfidences(data):
    print("Loading Confidences")
    loadMetric(data, metricName="confidence", metricFile="confidenceForResponsibleCell.txt")

def loadMetric(data, metricName="confidence", metricFile="confidenceForResponsibleCell.txt", forceDatasetLoadFolders=["Training"], detectMemberDataFolders=False):
    for sim in data.sims:
        for trial in sim.trials:
            if  not hasattr(trial.data, 'datasetMetrics') or trial.data.datasetMetrics is None:
                trial.data.datasetMetrics = {}

            # load testing rewards for all environments
            if detectMemberDataFolders:
                for member in trial.config["members"]:
                    # need to set all environment types
                    if member[1] == "VocYoloTaskDataset" or member[1] == "VocYoloTaskDataset" or member[1] == "VocClassificationTaskDataset":
                        memberName = member[0]
                        if  memberName not in trial.data.datasetMetrics:
                            trial.data.datasetMetrics[memberName] = {}
                        filePath = trial.path + memberName + "/" + metricFile 
                        trial.data.datasetMetrics[memberName][metricName] = Utils.DataUtil.numpyDataQuickLoadEasy(filePath)

            for memberName in forceDatasetLoadFolders:
                if  memberName not in trial.data.datasetMetrics:
                    trial.data.datasetMetrics[memberName] = {}
                filePath = trial.path + memberName + "/" + metricFile 
                trial.data.datasetMetrics[memberName][metricName] = Utils.DataUtil.numpyDataQuickLoadEasy(filePath)