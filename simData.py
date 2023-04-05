import os
import pickle
import code

class FigureData:
    def __init__(self, FolderPath, Figure, Title):
        self.fig = Figure
        self.title = Title
        self.folderPath = FolderPath
    
    def saveFigure(self):
        fullPath = self.folderPath + self.title

        if not os.path.exists(os.path.dirname(fullPath)):
            os.makedirs(os.path.dirname(fullPath))

        filePath = "%s%s" % (self.folderPath, self.title)
        print("Saving %s"  % filePath)
        self.fig.savefig(filePath)

class FigureList():
    def __init__(self):
        self.figs = list()

    def clearFigs(self):
        del self.figs
        self.figs = list()

    def addFigure(self, figureFolderPath, fig, title):
        self.figs.append(FigureData(figureFolderPath, fig, title))

    def saveFigures(self):
        for fig in self.figs:
            fig.saveFigure()

class TrialData:
    def __init__(self):
        self.food = None # numpy array of food

        # all these point to the same numpy object
        # just different slices
        self.performance = None # numpy array of performance
        self.judgement = None

        self.inputHiddenWeights = None
        self.hiddenOutputWeights = None
        self.stages = None

class Trial:
    def __init__(self):
        number = None # simulation number
        path = None
        title = None
        self.data = TrialData()
        # Trial specific parameters
        # later this will be hooked up to read in all parameters from simulation
        self.parameters = { 
            "iterationsInEpoch" : 600,
            "epochsPerAeon" : 100,
            
            # Layer Dimmensions
            "widthInputLayer" : 7,
            "widthHiddenLayer"  : 28,
            "widthOutputLayer"  : 9,

            # simulation Duration
            "totalAeons" : 0   
        }
        self.__figures = FigureList()

    def clearFigs(self):
        self.__figures.clearFigs()

    def addFigure(self, fig, title):
        self.__figures.addFigure("%s/figures/" % (self.path), fig, title)
    
    def saveFigures(self):
        self.__figures.saveFigures()

    def getFigureFolderPathAndEnsureExists(self):
        fp = "%s../figures/" % (self.path)
        if not os.path.exists(fp):
                os.makedirs(fp)
        return fp        

class Simulation:
    def __init__(self):
        path = None
        title = None 
        self.trials = list()
        self.__figures = FigureList()

    def clearFigs(self):
        self.clearSimulationFigs()
        self.clearTrialFigs()

    def clearSimulationFigs(self):
        self.__figures.clearFigs()

    def clearTrialFigs(self):
        for trial in self.trials:
            trial.clearFigs()

    def deepCopyOfSimShallowCopyTrials(self):
        newSimObj = Simulation()
        newSimObj.path = self.path
        newSimObj.title = self.title
        newSimObj.trials = map(lambda x: self.trials[x], range(0,len(self.trials)))
        return newSimObj

    def addFigure(self, fig, title):
        self.__figures.addFigure("%sfigures/" % (self.path), fig, title)

    def saveFigures(self):
        self.__figures.saveFigures()        

class SimData:
    def __init__(self, figureFolderPath=None, title=""):
        self.sims = list()
        self.figureFolderPath = figureFolderPath
        self.__figures = FigureList()
        self.titlePattern = None
        self.title = title

    def clearFigs(self):
        self.clearSimDataFigs()
        self.clearSimulationFigs()

    def clearSimDataFigs(self):
        self.__figures.clearFigs()

    def clearSimulationFigs(self):
        for sim in self.sims:
            sim.clearFigs()

    def addFigure(self, fig, title):
        self.__figures.addFigure("%sfigures/" % (self.figureFolderPath), fig, title)

    def printSimsPaths(self):
        for sim in self.sims:
            print(sim.path)
            for trial in sim.trials:
                print("       " + trial.path)
# Ways to Load Sim Data
    #recursive helper to create all file paths from pattern
    def _createSimulationStructureFromPatternHelper(self, depth, filePattern, titlePattern,listOfVariables , trials, removeWhiteSpaceFromPath=False):
        if len(listOfVariables) == 0: #if nothing else in list then all substitutions are complete
            sim = Simulation()
            sim.path = filePattern
            sim.title = titlePattern
            
            for i in trials:
                trial = Trial()
                trial.number = i
                trial.path = sim.path + "/trial" + str(i) + "/"
                if removeWhiteSpaceFromPath:
                    trial.path = trial.path.replace(" ","")
                trial.title = sim.title + " " + "Trial " + str(i)
                sim.trials.append(trial)

            self.sims.append(sim)
            return

        #if previous if statement is not hit the listOfVariables has at least one index in it
        for number in listOfVariables[0]:
            newPattern = filePattern.replace(str("{" + str(depth) + "}"), str(number))
            netTitlePattern = titlePattern.replace(str("{" + str(depth) + "}"), str(number))
            self._createSimulationStructureFromPatternHelper(depth + 1, newPattern, netTitlePattern, listOfVariables[1:] , trials, removeWhiteSpaceFromPath=removeWhiteSpaceFromPath)

    # this function was created to make loading data from sweeps easier
    # it leverages the parameter values stored by the simSweep.py code and leverages custom classes implemented there
    def createSimulationStructureSweepFolder(self, pathToSweepFolder, titlePattern, titlePatternSameAsFilePattern=False):
        # this pickle file should be a list of ParamPathValueSet from the simData.py class from the simulation code
        if self.figureFolderPath is None:
            self.figureFolderPath = pathToSweepFolder + "/"
        self.paramPathValueSetDicts = pickle.load(open(pathToSweepFolder + "/paramPathValueSets.pkl", "rb"))

        filePattern = pathToSweepFolder + "/"
        listOfVariables = []
        trials = [0] # default to zero seed

        paramIdx = 0
        for paramPathValueSetDict in self.paramPathValueSetDicts:
            # if param id is included in the file name string then include
            if paramPathValueSetDict["includeParamId"]:
                if not (filePattern ==  "" or filePattern == pathToSweepFolder+"/"): # then there are already parameters in string
                    filePattern += "_"
                filePattern += "%s-{%s}" % (paramPathValueSetDict["paramId"],  str(paramIdx))
                paramIdx += 1
                # listOfVariables.append(paramPathValueSetDict["values"])
                listOfVariables.append(paramPathValueSetDict["prettyValues"])

            # if parameter is the seed, then include in seeds
            if paramPathValueSetDict["paramId"] == "seed" or paramPathValueSetDict["isSeed"]:
                # trials = paramPathValueSetDict["values"]
                trials = paramPathValueSetDict["prettyValues"]

        if titlePatternSameAsFilePattern:
            titlePattern = filePattern
        filePattern += "/"
        self.createSimulationStructureFromPattern(filePattern, titlePattern, listOfVariables , trials, removeWhiteSpaceFromPath=True)

    #pattern - pattern to get to a group. assumed to replace{n}. n starts from 0
    #listOfVariables - list of lists, variables to replace in pattern 
    #trials - number of trials
    def createSimulationStructureFromPattern(self, filePattern, titlePattern, listOfVariables , trials, removeWhiteSpaceFromPath=False):
        self.titlePattern = titlePattern
        self._createSimulationStructureFromPatternHelper(0, filePattern, titlePattern, list(listOfVariables) , trials, removeWhiteSpaceFromPath=removeWhiteSpaceFromPath)

    def clearSimulationData(self):
        self.sims = list()

    # sims - a list of integers that corespond to the sim index to add to the new SimData
    # trials - a list of integers that corespond to the sim index to add to the new SimData
    # makes dep copy of simulation objec
    # makes shallow copy of list of trial objects to save on space and computation time
    def getReferenceToFewSims(self, sims, trials):
        ret = SimData()

        #gets the desired deep copy sims with shallow copies of all trials
        ret.sims = map(lambda x: self.sims[x].deepCopyOfSimShallowCopyTrials(), sims)
            
        # gets desired shallow copies of trials for each deep copy of sim
        for sim in ret.sims:
            sim.trials = map(lambda x: sim.trials[x], trials)

        return ret

    def saveFigures(self):
        self.__figures.saveFigures()
        for sim in self.sims:
            sim.saveFigures()
            for trial in sim.trials:
                trial.saveFigures()