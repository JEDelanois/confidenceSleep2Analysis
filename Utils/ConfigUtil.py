import json


# individual config helpers

def addDefaultsToConfig(defaults, config):
    for key in defaults:
        if key not in config:
            config[key] = defaults[key]

def loadConfigFromJson(path):
    f = open(path, "r")
    ret = json.load(f)
    f.close()
    return ret

def saveConfigToJson(config, path):
    f = open(path, "w")
    json.dump(config, f, indent=3)
    f.close()


def getMemberSetWithName(config, name):
    try:
        for memberSet in config["members"]:
            if memberSet[0] == name:
                return memberSet
    except expression as identifier:
        print("Error in getMemberWithName")
        print(expression)
        exit()
        

# Simulation helpers
def loadConfigsForSimulations(data):
   for sim in data.sims:
        for trial in sim.trials:
            configPath = trial.path + "config.json"
            trial.config = loadConfigFromJson(configPath)


