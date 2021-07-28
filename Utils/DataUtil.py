import numpy as np
import os
import sys

# used to help quickly load numpy objecs by using a .npy format
# does not implement loading of regular data because of possiblilty of loading comllexity
# ie combining multiply data files into single numpy array
# filepath - file path to expected np file
# expectedSize - expected size of numpy array. errors if size mismatch
# returns (success, numpyArray) 
#   success - bool if load was sucessful
#   numpyArray - data loaded                
def numpyDataQuickLoad(filePath, expectedSize):
    exists = os.path.isfile(filePath)
    # if the file exists
    if exists:
        try:
            data = np.load(filePath)
        except:
            print( "Error in numpyDataQuickLoad while loading file: %s" % filePath)
            exit()

        # if there is an expected size
        if expectedSize != None:

            # if expected size matches then return data
            if data.shape == expectedSize:
                return (True, data)
            # if expected size does not match there is a problem
            else:
                print("Error in numpyDataQuickLoad() for file %s expected size does not equal actual size" % filePath)
                print("expected size:")
                print(expectedSize)
                print("actual size:")
                print(data.shape)
                exit()
        else:
            return (True, data)
    # else the file does not exist
    else:
        return (False, None)



# used to help quickly load numpy objecs by using a .npy format
# filepath - file path to regular file
# expectedSize - expected size of numpy array. errors if size mismatch

# returns numpyArray
#   numpyArray - data loaded  returns None if error
def numpyDataQuickLoadEasy(filePath, expectedSize=None, forceTxtLoad=False):
    quickLoadFilePath = filePath + ".npy"
    if os.path.isfile(quickLoadFilePath) and not forceTxtLoad:
        try:
            data = np.load(quickLoadFilePath)
        except:
            print("Error in numpyDataQuickLoad while loading file: %s" % quickLoadFilePath)
            exit()

        # if there is an expected size
        if expectedSize != None:

            # if expected size matches then return data
            if data.shape == expectedSize:
                print("Qucikloaded file %s" % quickLoadFilePath)
                return data
            # if expected size does not match there is a problem
            else:
                print("1 Error in numpyDataQuickLoadEasy() for file %s expected size does not equal actual size" % quickLoadFilePath)
                print("expected size:")
                print(expectedSize)
                print("actual size:")
                print(data.shape)
                exit()
        else:
            print("Qucikloaded file %s" % quickLoadFilePath)
            return data



    # if the file exists
    if os.path.isfile(filePath):
        try:
            data = np.loadtxt(filePath)
        except:
            print("Error in numpyDataQuickLoad while loading file: %s" % filePath)
            exit()

        # if there is an expected size
        if expectedSize != None:

            # if expected size matches then return data
            if data.shape == expectedSize:
                try:
                    np.save(quickLoadFilePath, data)
                except:
                    print("Was not able to save file %s" % quickLoadFilePath)
                print("Loaded and quicksaved file %s" % filePath)
                return data
            # if expected size does not match there is a problem
            else:
                print("2 Error in numpyDataQuickLoad() for file %s expected size does not equal actual size" % filePath)
                print("expected size:")
                print(expectedSize)
                print("actual size:")
                print(data.shape)
                exit()
        else:
            try:
                np.save(quickLoadFilePath, data)
            except:
                print("Was not able to save file %s" % quickLoadFilePath)
            print("Loaded and quicksaved file %s" % filePath)
            return data
    else:
        print("File %s does not exist" % filePath)
        exit()
        return None
    return None