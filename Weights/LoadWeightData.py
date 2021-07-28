import torch
import Utils.ConfigUtil
import os
import Utils.GeneralUtil

# weight shape trial.weights[layer][epoch, post, pre] 
def loadWeightData(data):
    print("Loading Weights")
    for sim in data.sims:
         for trial in sim.trials: 
             weightFolderPath = trial.path + "weights/"
             print("Loading weights for simulation %s" % trial.path)
             modelMemberSet = Utils.ConfigUtil.getMemberSetWithName(trial.config, "model")
             modelParameters = modelMemberSet[2]
             numLayers = len(modelParameters["layers"])

             trial.weights = []
             for lay in range(numLayers-1):
                 layerWeights = []
                 layerFolderPath = weightFolderPath + "%dto%d-weights/" % (lay, lay+1)
                 quicksavePath = layerFolderPath + "weights.qs"
                
                 # quickload weights if present
                 if os.path.exists(quicksavePath):
                     print("    Quick loading layer %d to %d weights" % (lay, lay+1))
                     layerWeights = torch.load(quicksavePath)
                     trial.weights.append(layerWeights)

                 # load all weights
                 else:
                     print("    Reg loading layer %d to %d weights" % (lay, lay+1))
                     weightPaths = [layerFolderPath + fpath for fpath in os.listdir(layerFolderPath) if fpath.endswith(".pt")]
                     weightPaths.sort(key=Utils.GeneralUtil.natural_keys)
                     for weightPath in weightPaths:
                         weight = torch.load(weightPath).unsqueeze(0)
                         layerWeights.append(weight)
                     layerWeights = torch.cat(layerWeights, 0).detach().numpy()
                     trial.weights.append(layerWeights)
                     torch.save(layerWeights, quicksavePath)
