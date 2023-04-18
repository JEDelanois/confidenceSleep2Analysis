# code from https://colab.research.google.com/drive/1H_XlTbNvjxlAXMW5NuBDWhxF3F2Osg1F?usp=sharing#scrollTo=XldU3fNbOuSu
from torch.utils.data import DataLoader
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
# import tqdm
import torch.optim as optim
import torch.nn as nn
import matplotlib.patches as mpatches
# import Utils.OutputUtil as OutputUtil
import os
import code


# Use kwags for calibration method specific parameters
# TODO need to make num_classes configurable
def test(model, dataSet, calibration_method=None, dataLoaderParams={},**kwargs):
  preds = []
  labels_oneh = []
  correct = 0
  trainingMode = model.training
  model.eval()
  # TODO add data loader paramsgg
  test_loader = DataLoader(dataSet, **dataLoaderParams)
  with torch.no_grad():
      for [images, labels] in test_loader:
      # for data in tqdm(test_loader):
          # images, labels = data[0].to('cuda:0'), data[1].to('cuda:0')

          pred = model(images)
          
          if calibration_method:
            pred = calibration_method(pred, kwargs)

          # Get softmax values for model input and resulting class predictions
          sm = nn.Softmax(dim=1)
          pred = sm(pred)
          # print("ece preds")
          # print(pred)
          # print(pred.size())

          _, predicted_cl = torch.max(pred.data, 1)
          pred = pred.cpu().detach().numpy()

          # Convert labels to one hot encoding
          label_oneh = torch.nn.functional.one_hot(labels, num_classes=dataSet.numberClasses)
          label_oneh = label_oneh.cpu().detach().numpy()

          preds.extend(pred)
          labels_oneh.extend(label_oneh)

          # Count correctly classified samples for accuracy
          correct += sum(predicted_cl == labels).item()

  preds = np.array(preds).flatten()
  labels_oneh = np.array(labels_oneh).flatten()

  correct_perc = correct / len(dataSet)
  # print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct_perc))
  # print(correct_perc)
  print('ECE accuracy: %d %%' % (100 * correct_perc))

  # if model was in training mode, then restore to training mode
  if trainingMode:
      model.train()
  
  return preds, labels_oneh, correct_perc




def calc_bins(preds, labels_oneh):
  # Assign each prediction to a bin
  num_bins = 10
  bins = np.linspace(0.1, 1, num_bins)
  binned = np.digitize(preds, bins)

  # Save the accuracy, confidence and size of each bin
  bin_accs = np.zeros(num_bins)
  bin_confs = np.zeros(num_bins)
  bin_sizes = np.zeros(num_bins)

  for bin in range(num_bins):
    bin_sizes[bin] = len(preds[binned == bin])
    if bin_sizes[bin] > 0:
      bin_accs[bin] = (labels_oneh[binned==bin]).sum() / bin_sizes[bin]
      bin_confs[bin] = (preds[binned==bin]).sum() / bin_sizes[bin]

  return bins, binned, bin_accs, bin_confs, bin_sizes


def get_metrics(preds, labels_oneh):
  ECE = 0
  MCE = 0
  bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(preds, labels_oneh)

  for i in range(len(bins)):
    abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
    ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
    MCE = max(MCE, abs_conf_dif)

  return ECE, MCE

def getEceMce(model, dataSet, calibration_method=None, dataLoaderParams={}, **kwargs):
  preds, labels_oneh, ecePercentCorrect = test(model, dataSet,dataLoaderParams=dataLoaderParams, calibration_method=calibration_method, **kwargs)
  ECE, MCE = get_metrics(preds, labels_oneh)
  draw_reliability_graph(preds, labels_oneh,ECE, MCE, model)
  return ECE, MCE, ecePercentCorrect

# preds - numpy array of softmax falues for predictions 
# lables_oneh - numpy array of onehot labels
def getEceMceFromPreds(model, preds, labels_oneh, filePath, **kwargs):
  # origional code flattens these but code should work either way
  preds = preds.flatten()
  labels_oneh = labels_oneh.flatten()

  ECE, MCE = get_metrics(preds, labels_oneh)
  draw_reliability_graph(preds, labels_oneh,ECE, MCE, model, filePath=filePath)
  return ECE, MCE


def draw_reliability_graph(preds, labels_oneh,ECE, MCE, model, filePath=None):
  bins, binned, bin_accs, bin_confs, bin_sizes = calc_bins(preds, labels_oneh)
  # bins, _, bin_accs, _, _ = calc_bins(preds, labels_oneh)

  fig = plt.figure(figsize=(8, 8))
  ax = fig.gca()

  # x/y limits
  ax.set_xlim(0, 1.05)
  ax.set_ylim(0, 1)

  # x/y labels
  plt.xlabel('Confidence')
  plt.ylabel('Accuracy')

  # Create grid
  ax.set_axisbelow(True) 
  ax.grid(color='gray', linestyle='dashed')

  # Error bars
  plt.bar(bins, bins,  width=0.1, alpha=0.3, edgecolor='black', color='r', hatch='\\')

  # Draw bars and identity line
  plt.bar(bins, bin_accs, width=0.1, alpha=1, edgecolor='black', color='b')
  plt.plot([0,1],[0,1], '--', color='gray', linewidth=2)

  # Equally spaced axes
  plt.gca().set_aspect('equal', adjustable='box')

  # ECE and MCE legend
  ECE_patch = mpatches.Patch(color='green', label='ECE = {:.2f}%'.format(ECE*100))
  MCE_patch = mpatches.Patch(color='red', label='MCE = {:.2f}%'.format(MCE*100))
  plt.legend(handles=[ECE_patch, MCE_patch])

  #plt.show()
  
  if filePath is None:
    pass
    # calibratedNetworkPlotFolder = OutputUtil.outputLogger.ensureOutputDirectory("/figures/eceCalibration/")
    # plt.savefig('%s/calibrated_network_%d.png'% (calibratedNetworkPlotFolder, model.epochsTrained), bbox_inches='tight')
  else:
    os.makedirs(os.path.dirname(filePath), exist_ok=True)
    print("Saving figure %s" % filePath)
    plt.savefig(filePath, bbox_inches='tight')
  
  fig = plt.figure(figsize=(8, 8))
  ax = fig.gca()
  plt.bar(bins, bin_sizes,  width=0.1, edgecolor='black', color='b')
  plt.yscale("log")
  plt.ylabel("Bin Sizes")
  fp = os.path.join(os.path.dirname(filePath), "%s-%s" % ("binCounts", os.path.basename(filePath)))
  print("Saving figure %s" % fp)
  plt.savefig( fp, bbox_inches='tight')

  fig = plt.figure(figsize=(8, 8))
  ax = fig.gca()
  plt.bar(bins, bin_confs,  width=0.1, edgecolor='black', color='b')
  plt.ylabel("Bin Confidences")
  fp = os.path.join(os.path.dirname(filePath), "%s-%s" % ("binCofs", os.path.basename(filePath)))
  print("Saving figure %s" % fp)
  plt.savefig( fp, bbox_inches='tight')
  plt.close(fig)
