'''
Fluid script intended to change
Given the up front cost of computing all the CSV for classify, this script denecessitates regenerating the data in the event of a minor formatting 
Update: Now it applies all kinds of manipulations to the data after initial generation

Args
    --ws
        Takes every CSV passed and outputs the whitespace and observations in each file
    --formatWs <folder>
        Takes the folder and adds whitespace to the end of each data file
            For some reason, the classify app needs this
        Assumes folder hierarchy of <folder>/(labels/ & features/ & classModelData/)
    --formatL
        Takes folder and separates observation format of line pairs into observation per line
    --binaryMap
        Take folder hierarchy and convert classifications to binary
    --normalize
        Take folder hierarchy and normalize ecg node vals for all data
    --SMOTE
        Smote synthesized minority oversampling. Currently only supports binary MLII data
    --denseOversample
        SMOTE inspired method for oversampling variable length data
        pass t or f for 2D vs 1D
        pass int (1-100) for proportion of oversampled to produce relative to majority class, int represents % of majority class
            NEED TO ADD SUPPORT FOR QUINARY CLASSIFICATION
    --denseUndersample
        Stochastic undersampling for variable length data
        pass t or f for 2D vs 1D
        pass int for (1-100), removes that % of data
    --proportion
        % of dataset to make the minority label
    --seed
        int to seed randomizer
    --OP
        Path to data on which to act
    --log_scalar
        Logarithmic scaling applied to data. pass t or f for 2D vs 1D
    --z_score
        z score applied to data. pass t or f for 2D vs 1D
    --1d
        make a 2d node directory 1 node data
NOTE: need to update this to support running multiple options concurrently
    feature creep has not been accounted for
Also file system traversal can be abstracted for several functions with common operations
'''
import csv 
import sys
import os
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import scipy.stats as stats
from csv_to_training import normalizeNumVec
import random

def getCmdArgs():
    csvs = []
    opts = {
        "formatWs": False,
        "ws": False,
        "formatL": False,
        "binaryMap": False,
        "OP": False,
        "SMOTE": False,
        "normalize": False,
        "log_scalar": False,
        "z_score": False,
        "denseOversample": False,
        "denseUndersample": False,
        "proportion": False,
        "seed": False,
        "scalar": False,
        "1d": False,
    }
    for i, arg in enumerate(sys.argv[1:]):
        if ".csv" in arg:
            csvs.append(arg)
        elif arg == '--formatWs':
            opts["formatWs"] = True
            opts["formatPath"] = sys.argv[i + 2]
        elif arg == '--ws':
            opts["ws"] = True
        elif arg == '--formatL':
            opts["formatL"] = True
            opts["formatPath"] = sys.argv[i + 2]
        elif arg == "--binaryMap":
            opts["binaryMap"] = True
            opts["OP"] = sys.argv[i + 2]
        elif arg == "--SMOTE":
            opts["SMOTE"] = True
            opts["OP"] = sys.argv[i + 2]
        elif arg == "--denseOversample":
            opts["denseOversample"] = True
            opts["multi_d"] = (sys.argv[i+2] == 't')
            opts["proportion"] = int(sys.argv[i+3])
            opts["scalar"] = int(sys.argv[i+4])
        elif arg == "--denseUndersample":
            opts["denseUndersample"] = True
            opts["multi_d"] = (sys.argv[i+2] == 't')
            opts["proportion"] = int(sys.argv[i+3])
        elif arg == "--normalize":
            opts["normalize"] = True
            opts["OP"] = sys.argv[i + 2]
        elif arg == "--log_scalar":
            opts["log_scalar"] = True
            opts["multi_d"] = (sys.argv[i+2] == 't')
        elif arg == "--z_score":
            opts["z_score"] = True
            opts["multi_d"] = (sys.argv[i+2] == 't')
        elif arg == "--OP":
            opts["OP"] = sys.argv[i + 2]
        elif arg == "--seed":
            opts["seed"] = int(sys.argv[i + 2])
        elif arg == "--1d":
            opts["1d"] = True
            opts["OP"] = sys.argv[i + 2]


    return csvs, opts

def reduceDimensionality(parent_folder):
    folders = ["features", "classModelData"]
    for folder in folders:
        files = os.listdir(f"{parent_folder}/{folder}")
        for file in files:
            if("Data.csv" in file):       
                rows = []
                with open(f"{parent_folder}/{folder}/{file}", 'r') as f:
                    read = csv.reader(f)
                    for i, row in enumerate(read):
                        if(i % 2 == 0):
                            rows.append(row)
                with open(f"{parent_folder}/{folder}/{file}", 'w') as f:
                    write = csv.writer(f)
                    write.writerows(rows)

def validateWhitespace(csvs):
    whitespace = 0
    observations = 0
    for i, file in enumerate(csvs):
        with open(file, "r") as f:
            reader = csv.reader(f)
            for line in reader:
                if line == []:
                    whitespace += 1
                else:
                    observations += 1
        print(f"Whitespace in file {i}: {whitespace}\nObservations (populated lines): {observations}")
        whitespace = 0
        observations = 0

def formatFolderWs(parent_folder):
    folders = ["features", "classModelData"]
    for folder in folders:
        files = os.listdir(f"{parent_folder}/{folder}")
        for file in files:
            if("Data.csv" in file):
                with open(f"{parent_folder}/{folder}/{file}", 'a') as f:
                    f.write('\n')

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

def formatFolderL(parent_folder):
    folders = ["features", "classmodeldata"]
    for folder in folders:
        files = os.listdir(f"{parent_folder}/{folder}")
        for file in files:
            if("data.csv" in file):
                rows = []
                with open(f"{parent_folder}/{folder}/{file}", 'r') as f:
                    read = csv.reader(f)
                    #print(len(list(read)))
                    for line in read:
                        sample = 0    
                        tmp = ""
                        print(len(line))
                        #exit()
                        for el in line:
                            '''if sample and sample % 3 == 0:
                                rows.append([])
                                continue
                            '''
                            if(sample % 3 == 0 and sample):
                                rows.append([])

                            elif(not has_numbers(el)):
                                print("printing in temp")
                                #print(tmp)

                                #exit()

                                rows.append([tmp[:-1]])
                                tmp = ""
                            else:
                                tmp = tmp + el + ','
                           #sample += 1
                            sample += 1
                    print("printing rows")
                    print(rows)
                    exit()
                writeFileL(f"{parent_folder}/{folder}/{file}", rows)
                        #rows.append()
                        #print(line)
                        #exit()
                        #if(line != ", "):
                        #tmp = tmp + line
def writeFileL(path, rows):
    with open(path, "w") as f:
        write = csv.writer(f)
        write.writerows(rows)

def binaryMap(parent_folder):
    folders = ["labels", "classModelData"]
    for folder in folders:
        files = os.listdir(f"{parent_folder}/{folder}")
        for file in files:
            if("Labels.csv" in file):
                rows = []
                with open(f"{parent_folder}/{folder}/{file}", 'r') as f:
                    read = csv.reader(f)
                    #print(len(list(read)))
                    for line in read:
                        if(int(line[0]) >= 1):
                            rows.append([1])
                        else:
                            rows.append([0])
                writeFileL(f"{parent_folder}/{folder}/{file}", rows)

def normalizeSet(parent_folder):
    folders = ["features", "classModelData"]
    for folder in folders:
        files = os.listdir(f"{parent_folder}/{folder}")
        for file in files:
            if("Data.csv" in file):
                rows = []
                with open(f"{parent_folder}/{folder}/{file}", 'r') as f:
                    read = csv.reader(f)
                    for line in read:
                        row = []
                        numLine = [int(x) for x in line]
                        for entry in line:
                            row.append(normalizeNumVec(int(entry), numLine))
                        rows.append(row)
                writeFileL(f"{parent_folder}/{folder}/{file}", rows)

def log_scaling(parent_folder, is2D):
    folders = ["features", "classModelData"]
    for folder in folders:
        files = os.listdir(f"{parent_folder}/{folder}")
        for file in files:
            if("Data.csv" in file):
                sampleSets = []
                with open(f"{parent_folder}/{folder}/{file}", 'r') as f:
                    read = csv.reader(f)
                    read = list(read)
                    if(is2D):
                        for i in range(0, len(read), 2):
                            col1, col2 = [], []
                            for k in range(len(read[i])):
                                col1.append(int(read[i][k]))
                                col2.append(int(read[i+1][k]))
                            #print(col1)
                            col1 = (np.log(col1)).tolist()
                            col2 = (np.log(col2)).tolist()
                            sampleSets.append(col1)
                            sampleSets.append(col2)
                    else:
                        for i in range(0, len(read)):
                            col1 = []
                            for k in range(len(read[i])):
                                col1.append(int(read[i][k]))
                            col1 = (np.log(col1)).tolist()
                            sampleSets.append(col1)

                writeFileL(f"{parent_folder}/{folder}/{file}", sampleSets)

def z_score(parent_folder, is2D):
    folders = ["features", "classModelData"]
    for folder in folders:
        files = os.listdir(f"{parent_folder}/{folder}")
        for file in files:
            if("Data.csv" in file):
                sampleSets = []
                with open(f"{parent_folder}/{folder}/{file}", 'r') as f:
                    read = csv.reader(f)
                    read = list(read)
                    if(is2D):
                        for i in range(0, len(read), 2):
                            col1, col2 = [], []
                            for k in range(len(read[i])):
                                col1.append(int(read[i][k]))
                                col2.append(int(read[i+1][k]))
                            #print(col1)
                            col1 = (stats.zscore(np.array(col1)))
                            col2 = (stats.zscore(np.array(col2)))
                            sampleSets.append(col1)
                            sampleSets.append(col2)
                    else:
                        for i in range(0, len(read)):
                            col1 = []
                            for k in range(len(read[i])):
                                col1.append(int(read[i][k]))
                            col1 = (stats.zscore(np.array(col1)))
                            sampleSets.append(col1)

                writeFileL(f"{parent_folder}/{folder}/{file}", sampleSets)



def denseUndersample(parent_folder, is2D, proportion):
    folders = ["features"]#, "classModelData"]
    for folder in folders:
        files = os.listdir(f"{parent_folder}/{folder}")
        for file in files:
            if("Data.csv" in file):
                filenameX = ""
                filenameY = ""
                if("class" in file):
                    filenameX = f"{parent_folder}/{folder}/classData.csv"
                    filenameY = f"{parent_folder}/{folder}/classLabels.csv"
                else:
                    filenameX = f"{parent_folder}/{folder}/{file}"
                    filenameY = f"{parent_folder}/labels/patient{file[-11:-8]}Labels.csv"
                #rows = []
                #with open(f"{parent_folder}/{folder}/{file}", 'r') as f:
                maxLen = 0
                with open(f"{parent_folder}/{folder}/{file}", 'r') as f:
                    read = csv.reader(f)
                    for row in read:
                        if(len(row) > maxLen):
                            maxLen = len(row)
                maxCols = [x for x in range(maxLen)]
                print(f"filename: {filenameX}")
                X = pd.read_csv(filenameX, names=maxCols)
                y = pd.read_csv(filenameY)

                majorityLabel = []
                minorityLabel = []

                #extract row position of classifications
                for i, row in y.iterrows():
                    if(row[0] == 0):
                        majorityLabel.append(i)
                    else:
                        minorityLabel.append(i)

                #for 2D or 1D, downselect DFs to minority label rows only
                X_MLII_Majority = []
                X_Vnum_Majority = []
                Y_majority = y.iloc[majorityLabel]
                delListX = []
                delListY = []
                if(is2D):
                    X_MLII_Majority = [0 if x == 0 else x*2+1 for x in majorityLabel]
                    X_Vnum_Majority = [1 if x == 0 else x*2+2 for x in majorityLabel]
                
                    X_MLII_MajorityDF = X.iloc[X_MLII_Majority]
                    X_Vnum_MajorityDF = X.iloc[X_Vnum_Majority]
                    
                    numUndersample = int((proportion/100) * len(majorityLabel))
                    print(f"majority entries {len(majorityLabel)}\ninstances to remove {numUndersample}")
                    
                    for i in range(numUndersample):
                        dropIndX = random.choice([X_MLII_Majority[i] for i in range(0, len(majorityLabel)) if X_MLII_Majority[i] not in delListX])
                        dropIndY = majorityLabel[X_MLII_Majority.index(dropIndX)]
                        delListX.append(dropIndX)
                        delListX.append(dropIndX+1)
                        delListY.append(dropIndY)
                    
                else:
                    X_MLII_Majority = majorityLabel
                
                    X_MLII_MajorityDF = X.iloc[X_MLII_Majority]

                    
                    numUndersample = int(proportion * len(majorityLabel))

                    #delete row pairs numUndersample times, randomly
                    
                    for i in range(numUndersample):
                        dropIndX = random.choice([X_MLII_Majority[i] for i in range(0, len(majorityLabel)) if X_MLII_Majority[i] not in delList])
                        dropIndY = majorityLabel(X_MLII_Majority.index(dropIndX))
                        delListX.append(dropIndX)
                        delListY.append(dropIndY)
                X.drop(X.index[delListX], inplace=True)
                y.drop(y.index[delListY], inplace=True)
                if(is2D):
                    targetLen = len(X.index) / 2
                else:
                    targetLen = len(X.index)
                
                while(len(y.index) < (targetLen)):
                    print(f"Len of label file error on {filenameY}, populating with prev class\n")
                    y = pd.concat([y, y.iloc[[len(y.index)-1]]])
                X.to_csv(filenameX, index=False, header=False, na_rep='0')
                y.to_csv(filenameY, index=False, header=False, na_rep='0')
                targetLen = None
                
                rows = []
                with open(filenameX, 'r') as f:
                    read = csv.reader(f)
                    for row in read:
                        row = [x for x in row if x != '0' and x != 0]
                        rows.append(row)
                with open(filenameX, 'w') as f:
                    write = csv.writer(f)
                    write.writerows(rows)
                
                            

#could apply knn to 1d data, single column
def denseOversample(parent_folder, is2D, proportion, scalar):
    folders = ["features"] #probably need to apply this function per patient, then aggregate for class model data, "classModelData"]
    for folder in folders:
        files = os.listdir(f"{parent_folder}/{folder}")
        for file in files:
            if("Data.csv" in file):
                filenameX = ""
                filenameY = ""
                if("class" in file):
                    filenameX = f"{parent_folder}/{folder}/classData.csv"
                    filenameY = f"{parent_folder}/{folder}/classLabels.csv"
                else:
                    filenameX = f"{parent_folder}/{folder}/{file}"
                    filenameY = f"{parent_folder}/labels/patient{file[-11:-8]}Labels.csv"
                maxLen = 0
                with open(f"{parent_folder}/{folder}/{file}", 'r') as f:
                    read = csv.reader(f)
                    for row in read:
                        if(len(row) > maxLen):
                            maxLen = len(row)
                maxCols = [x for x in range(maxLen)]
                print(f"filename: {filenameX}")
                #pandas requires this because it won't implicitly read variable row length CSVs
                X = pd.read_csv(filenameX, names=maxCols)
                y = pd.read_csv(filenameY)

                majorityLabel = []
                minorityLabel = []

                #extract row position of classifications
                for i, row in y.iterrows():
                    if(row[0] == 0):
                        majorityLabel.append(i)
                    else:
                        minorityLabel.append(i)
                #need to account for the fact that there are twice as many entries for 2D set while computing observation num

                #for 2D or 1D, downselect DFs to minority label rows only
                X_MLII_Minority = []
                X_Vnum_Minority = []
                Y_minority = y.iloc[minorityLabel]
                if(is2D):
                    X_MLII_Minority = [0 if x == 0 else x*2+1 for x in minorityLabel]
                    X_Vnum_Minority = [1 if x == 0 else x*2+2 for x in minorityLabel]
                
                    X_MLII_MinorityDF = X.iloc[X_MLII_Minority]
                    X_Vnum_MinorityDF = X.iloc[X_Vnum_Minority]

                    #for 2D. get std dev of each timestep after rpeak for patient    
                    stdDevListMLII = []
                    stdDevListVnum = []
                    
                    prev = None #catching off by one error
                    for col in X_MLII_MinorityDF:
                        #it is unclear to me why the exception handling is ever necessary, but it's only ever an off by one error
                        try:
                            #ignores null columns for variable length timesteps
                            stdDevListMLII.append(X_MLII_MinorityDF[col].std())
                            prev = X_MLII_MinorityDF[col].std()
                        except IndexError as err:
                            stdDevListMLII.append(prev)
                            break
                    for col in X_Vnum_MinorityDF:
                        #it is unclear to me why the exception handling is ever necessary, but it's only ever an off by one error
                        try:
                            #ignores null columns for variable length timesteps
                            stdDevListVnum.append(X_Vnum_MinorityDF[col].std())
                            prev = X_Vnum_MinorityDF[col].std()
                        except IndexError as err:
                            stdDevListMLII.append(prev)
                            break
                    
                    #get x * minorityNum, where x is the multiplicative value to achieve proportion % relative to majorityNum
                    oversampledEntriesScalar = int((len(majorityLabel) * (proportion / 100)) / len(minorityLabel))

                    XperturbationsDf = pd.DataFrame()
                    YperturbationsDf = pd.DataFrame()
                    #iterate over minority data entries X times
                    for i in range(oversampledEntriesScalar):
                        #iterate for each minority data entry
                        for k in range(len(X_MLII_Minority)):
                            #select entry in observation set randomly
                            observationNum = random.randrange(len(minorityLabel))
                            X_MLII_Perturbed = X_MLII_MinorityDF.iloc[[observationNum]].copy()
                            X_Vnum_Perturbed = X_Vnum_MinorityDF.iloc[[observationNum]].copy()

                            #perturbation in same direction for each observation
                            #perturbation product of float from vector [-1, 1] chosen from random uniform range multiplied by std dev at that time step
                            perturbation = random.uniform(-1, 1)
                            #guarantee perturbation is non-zero
                            while(perturbation == 0):
                                perturbation = random.uniform(-1, 1)
                            #iterate for each column in randomly selected row
                            for j in range(len(X_MLII_Perturbed.columns)):
                                #apply perturbation at each timestep of observation
                                if(np.isnan(X_MLII_Perturbed.iloc[0][j])):
                                    break
                                mliiStdDev = None
                                if(stdDevListMLII[j] == 0):
                                    print(f"std dev mlii at col {j} is 0, replacing with 1")
                                    mliiStdDev = 1
                                else:
                                    mliiStdDev = stdDevListMLII[j]
                                X_MLII_Perturbed.at[0, j] = X_MLII_Perturbed.iloc[0][j] + ((perturbation * mliiStdDev) * scalar)
                                #perturbation scalar applied uniformly to std dev of MLII AND V#
                                VnumStdDev = None
                                if(stdDevListVnum[j] == 0):
                                    print(f"std dev vnum at col {j} is 0, replacing with 1")
                                    VnumStdDev = 1
                                else:
                                    VnumStdDev = stdDevListVnum[j]
                                X_Vnum_Perturbed.at[0, j] = X_MLII_Perturbed.iloc[0][j] + ((perturbation * VnumStdDev) * scalar)
                            try:
                                XperturbationsDf = pd.concat([XperturbationsDf, X_MLII_Perturbed.iloc[[1]], X_Vnum_Perturbed.iloc[[1]]])
                                label = Y_minority.iloc[[observationNum]].copy()
                                YperturbationsDf = pd.concat([YperturbationsDf, label])
                            except IndexError as err:
                                print(f"bounds error on these DFs:\n{X_MLII_Perturbed}\n{X_Vnum_Perturbed}\nBreaking early for file {filenameX}")
                                break
                            
                    X = pd.concat([X, XperturbationsDf])
                    y = pd.concat([y, YperturbationsDf])
                    while(len(y.index) < (len(X.index) / 2)):
                        print(f"Len of label file error on {filenameY}, populating with anomalous observation\n")
                        y = pd.concat([y, y.iloc[[len(y.index)-1]]])
                    X.to_csv(filenameX, index=False, header=False, na_rep='0')
                    y.to_csv(filenameY, index=False, header=False, na_rep='0')
                    rows = []
                    with open(filenameX, 'r') as f:
                        read = csv.reader(f)
                        for row in read:
                            row = [x for x in row if x != '0' and x != 0]
                            rows.append(row)
                    with open(filenameX, 'w') as f:
                        write = csv.writer(f)
                        write.writerows(rows)
                            


                else:
                    X_MLII_Minority = minorityLabel
                
                    X_MLII_MinorityDF = X.iloc[X_MLII_Minority]

                    stdDevListMLII = []
                    
                    prev = None #catching off by one error
                    for col in X_MLII_MinorityDF:
                        #it is unclear to me why the exception handling is ever necessary, but it's only ever an off by one error
                        try:
                            #ignores null columns for variable length timesteps
                            stdDevListMLII.append(X_MLII_MinorityDF[col].std())
                            prev = X_MLII_MinorityDF[col].std()
                        except IndexError as err:
                            stdDevListMLII.append(prev)
                            break
                    
                    #get x * minorityNum, where x is the multiplicative value to achieve proportion % relative to majorityNum
                    oversampledEntriesScalar = int((len(majorityLabel) * (proportion / 100)) / len(minorityLabel))

                    XperturbationsDf = pd.DataFrame()
                    YperturbationsDf = pd.DataFrame()
                    #iterate over minority data entries X times
                    for i in range(oversampledEntriesScalar):
                        #iterate for each minority data entry
                        if(len(X_MLII_Minority) != len(stdDevListMLII)):
                            print(f"std dev list for file {filenameX} not fully populated")
                        for k in range(len(stdDevListMLII)): #should be equivalent to len X_MLII_Minority, again off by one eerror
                            #select entry in observation set randomly
                            observationNum = random.randrange(len(minorityLabel))
                            X_MLII_Perturbed = X_MLII_MinorityDF.iloc[[observationNum]].copy()

                            #perturbation in same direction for each observation
                            #perturbation product of float from vector [-1, 1] chosen from random uniform range multiplied by std dev at that time step
                            perturbation = random.uniform(-1, 1)
                            #guarantee perturbation is non-zero
                            while(perturbation == 0):
                                perturbation = random.uniform(-1, 1)
                            #iterate for each column in randomly selected row
                            for j in range(len(X_MLII_Perturbed.columns)):
                                #apply perturbation at each timestep of observation
                                if(np.isnan(X_MLII_Perturbed.iloc[0][j])):
                                    break
                                X_MLII_Perturbed.at[0, j] = X_MLII_Perturbed.iloc[0][j] + ((perturbation * stdDevListMLII[j]) * scalar)
                            try:
                                XperturbationsDf = pd.concat([XperturbationsDf, X_MLII_Perturbed.iloc[[1]]])
                                label = Y_minority.iloc[[observationNum]].copy()
                                YperturbationsDf = pd.concat([YperturbationsDf, label])
                            except IndexError as err:
                                print(f"bounds error on these DFs:\n{X_MLII_Perturbed}\nBreaking early for file {filenameX}")
                                break
                            
                    X = pd.concat([X, XperturbationsDf])
                    y = pd.concat([y, YperturbationsDf])
                    while(len(y.index) < (len(X.index))):
                        print(f"Len of label file error on {filenameY}, populating with anomalous observation\n")
                        y = pd.concat([y, y.iloc[[len(y.index)-1]]])
                    X.to_csv(filenameX, index=False, header=False, na_rep='0')
                    y.to_csv(filenameY, index=False, header=False, na_rep='0')
                    rows = []
                    with open(filenameX, 'r') as f:
                        read = csv.reader(f)
                        for row in read:
                            row = [x for x in row if x != '0' and x != 0]
                            rows.append(row)
                    with open(filenameX, 'w') as f:
                        write = csv.writer(f)
                        write.writerows(rows)
                    
                '''
                iteratively, following each annotation per patient represents x time since an r peak
                utilize this for the minority class to find the <mean line?> std dev for the samples at that timestep
                could randomly pick samples/timesteps to perturb, but may disrupt trends in data
                could instead uniformly apply for each sample
                either way, the intention is to take either samples from the set randomly or from the mean line
                    randomly select whole lines from dataset to perturb
                and add samples to the set with the perturbation per time step of randfloat(-1,1) * std dev
                can easily apply to 2D

                does this need to take place proportionally? specifying proportion of dataset, seed, etc
                iterate columns of non-zero label data. take std dev, apply above method to df
                for zero entries, can randomly apply undersampling. also in some form of ratio?
                '''

def sampling(parent_folder):
    folders = ["features", "classModelData"]
    for folder in folders:
        files = os.listdir(f"{parent_folder}/{folder}")
        for file in files:
            if("Data.csv" in file):
                filenameX = ""
                filenameY = ""
                if("class" in file):
                    filenameX = f"{parent_folder}/{folder}/classData.csv"
                    filenameY = f"{parent_folder}/{folder}/classLabels.csv"
                else:
                    filenameX = f"{parent_folder}/{folder}/{file}"
                    filenameY = f"{parent_folder}/labels/patient{file[-11:-8]}Labels.csv"
                #rows = []
                #with open(f"{parent_folder}/{folder}/{file}", 'r') as f:
                X = pd.read_csv(filenameX)
                y = pd.read_csv(filenameY)
                superset = SMOTE()#sampling_strategy=0.1)
                subset = RandomUnderSampler()#sampling_strategy=0.5)
                pipe = [('o', superset), ('u', subset)]
                transformer = Pipeline(steps=pipe)
                try:
                    X, y = transformer.fit_resample(X, y)
                    print(f"succeeded on {file}")
                except ValueError as err:
                    print(f"oversampling failed on {file}")
                    print(err)
                #    exit()
                X.to_csv(filenameX, header=False, index=False)
                y.to_csv(filenameY, header=False, index=False)

if __name__ == "__main__":
    csvs, opts = getCmdArgs()
    if(opts["seed"]):
        random.seed(opts["seed"])
    else:
        random.seed(0)
    if(opts["ws"]):
        validateWhitespace(csvs)
    if(opts["formatWs"]):
        formatFolderWs(opts["formatPath"])
    if(opts["formatL"]):
        formatFolderL(opts["formatPath"])
    if(opts["log_scalar"]):
        log_scaling(opts["OP"], opts["multi_d"])
    if(opts["z_score"]):
        z_score(opts["OP"], opts["multi_d"])
    if(opts["binaryMap"]):
        binaryMap(opts["OP"])
    if(opts["normalize"]):
        normalizeSet(opts["OP"])
    if(opts["SMOTE"]):
        sampling(opts["OP"])
    if(opts["denseOversample"]):
        denseOversample(opts["OP"], opts["multi_d"], opts["proportion"], opts["scalar"])
    if(opts["denseUndersample"]):
        denseUndersample(opts["OP"], opts["multi_d"], opts["proportion"])
    if(opts["1d"]):
        reduceDimensionality(opts["OP"])