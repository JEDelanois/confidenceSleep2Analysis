import sys
import pathlib
import importlib

def reloadLocalModules():
    for module in list(sys.modules.values()):
        try:
            if str(pathlib.Path().absolute()) in module.__file__:
                print(module)
                importlib.reload(module)
        except:
            temp = 1