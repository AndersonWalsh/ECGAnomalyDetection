'''
Assumes presence of populated csv_mitbih_database folder
Converts time from min:sec:ms to ms
Maps 15 classifications down to 5 by AAMI, as integers
0 (N): 
    normal beat, 
    left bundle branch block beat, 
    right bundle branch block beat
    atrial escape beat
    nodal escape beat
1 (S):
    atrial premature beat
    aberrated atrial premature beat
    nodal premature beat
    supraventricular premature beat
2 (V):
    premature ventricular contraction
    ventricular escape beat
3 (F):
    fusion of ventricular and normal beat
4 (Q):
    paced beat
    fusion of paced and normal beat
    unclassified beat
'''

import os
import pandas as pd
from collections import defaultdict

def getFiles(path, type="annot", negate=False):
    CSVs = []
    filenames = []
    for file in os.listdir(path):
        if(file == ".gitkeep"):
            continue
        if(not negate):
            if(type in file):
                CSVs.append(pd.read_csv(f"{path}/{file}"))
                filenames.append(file)
        else:
            if(type not in file):
                try:
                    CSVs.append(pd.read_csv(f"{path}/{file}"))
                    filenames.append(file)
                except pd.errors.EmptyDataError:
                    print("pandas error")
                    print(file)
    return CSVs, filenames

def readAnnots(path):
    return getFiles(path)

def readCSVs(path):
    return getFiles(path, negate=True)


def modAnnotVals(files):
    for file in files:
        file["Type"] = file["Type"].map(mapLabelSymbols()).fillna(4)
        file["Time"] = file["Time"].apply(timeConv)
        file["Type"] = file["Type"].apply(int)
    return files

def modFeatureNames(files):
    for file in files:
        file.rename(mapfeatureNames(), axis=1, inplace=True)
    return files

def writeDFs(path, files, filenames):
    for i in range(len(files)):
        files[i].to_csv(f"{path}/{filenames[i]}", index=False)

def timeConv(time):
    ms = int(time[time.rfind(".")+1:])
    ms += int(time[time.find(":")+1:time.rfind(".")]) * 1000
    ms += int(time[:time.find(":")]) * 60000
    return ms

def mapfeatureNames():
    featureMappings = {
        "'sample #'": 'Sample',

        "'MLII'": 0,

        "'V1'": 1,
        "'V2'": 2,
        "'V4'": 4,
        "'V5'": 5
    }
    return featureMappings

def mapLabelSymbols():
    symbolMappings = {
        '·': 0,
        'N': 0,
        'L': 0,
        'R': 0,
        'e': 0,
        'j': 0,

        'A': 1,
        'a': 1,
        'J': 1,
        'S': 1,

        'V': 2,
        'E': 2,

        'F': 3,

        '/': 4,
        'f': 4,
        'Q': 4
    }
    return symbolMappings

if __name__ == "__main__":
    path = './csv_mitbih_database'
    CSVs, filenames = readCSVs(path)
    writeDFs(path, modFeatureNames(CSVs), filenames)
    CSVs, filenames = readAnnots(path)
    writeDFs(path, modAnnotVals(CSVs), filenames)