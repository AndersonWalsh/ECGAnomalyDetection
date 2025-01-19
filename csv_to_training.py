'''
Assumes populated csv_mitbih_database
    each <int>.csv must have a corresponding annotations file
Produces formatted feature files and label files
    Removes unusable features
    Redistributes features between two files
    Puts observations onto single lines
Notably, time has to be removed from the label dataset and transferred to the feature set
    But, because the classify app takes a time value for each observation
        even if all the observations map to one classification
    Have to extrapolate time value for a given observation based on sample number
        Default data reflects that the time between each sample is uniform
        Compute time as a function of sample number for dataset
'''
#the problem is that the column labels aren't actually uniform. we need to understand the data better

import os
import pandas as pd
import csv_val_formatting as cvf
import csv
import sys

def getCmdArgs():
    dataOpts = {"ST": 0, "OP": './csv_mitbih_training', "D": 0, "V#": 1, "N": 0, "B": 0, "SR": 0}
    for i, arg in enumerate(sys.argv):
        try:
            if(arg == "--help"):
                print("Optional arguments)\n--ST (scale time): 0|1 , 1 - time scaling is continuous 0 - time scaling is proportional to respective sample range (starts at 0, per classification). Defaults to 0, 1 generates sparse data\n--OP (output path): path to a single directory. Assumed to have two subdirectories named 'features' and 'labels'. Defaults to './csv_mitbih_training'\n--D (dense): 0|1 produce dense data, defaults to 1\n--V#: 0|1, whether to use second ECG node, defaults to 1\n--N: 0|1, whether to normalize ECG readings (between 0 and 1 relative to min and max of col for patient) (defaults to 0)\n--B: 0|1, map class oriented specifier to binary classification problem (defaults to 0)\n--SR (scale by range): <num>, 0 - default time scaling takes place <int > 0> - scaling takes place uniformly based on +/- int on either side of each annotation")
                exit()
            match arg:
                case "--ST":
                    dataOpts["ST"] = int(sys.argv[i + 1])
                case "--OP":
                    dataOpts["OP"] = sys.argv[i + 1]
                case "--D":
                    dataOpts["D"] = int(sys.argv[i + 1])
                case "--V#":
                    dataOpts["V#"] = int(sys.argv[i + 1])
                case "--N":
                    dataOpts["N"] = int(sys.argv[i + 1])
                case "--B":
                    dataOpts["B"] = int(sys.argv[i + 1])
                case "--SR":
                    dataOpts["SR"] = int(sys.argv[i + 1])
        except:
            print("Error in commandline options. Run with --help")
            exit()
    return dataOpts

def normalizeNumVec(num, vector):
    return (num - min(vector)) / (max(vector) - min(vector))

def normalizeNum(num, minV, maxV):
    return (num - minV) / (maxV - minV)

def normalizeFeatures(patient):
    patientCols = patient.columns.values.tolist()
    patientCols.remove("Sample")
    for col in patientCols:
        minCol = patient[col].min()
        maxCol = patient[col].max()
        patient[col] = patient[col].apply(normalizeNum, minV=minCol, maxV=maxCol)
    return patient 

def readPatients(path):
    patients = {}
    for file in os.listdir(path):
        if(".csv" in file and "annot" not in file):
            patientNum = int(file[:len(file) - 4])
            patients[patientNum] = [path + '/' + file, path + '/' + file[:3] + 'annotations.csv']
            
    return patients

def readPatientsData(patients, dataOpts):
    for patient in patients.keys():
        patients[patient] = [pd.read_csv(patients[patient][0]), pd.read_csv(patients[patient][1])]
        if(dataOpts["N"]):
            patients[patient][0] = normalizeFeatures(patients[patient][0])
    return patients 

def removeDeadColumns(patients):
    for patient in patients.keys():
        patients[patient][1].drop(labels=["Time", "Sample","Sub", "Chan", "Num", "Aux"], inplace=True, axis=1)
    return patients

def moveTimeCol(patients):
    for patient in patients.keys():
        '''
        Constant scalar derived as a function of time/samples
            maps samples in training data to time
        '''
        classificationSamples = patients[patient][1]["Sample"]
        classificationTimes = patients[patient][1]["Time"]
        trainTimeCol = []
        classificationIter = 0
        #change algorithm to make individual sample ranges lines/dfs?
        for sample in patients[patient][0]["Sample"]:
            #last observation happens before last sample, often. Lazily prevents bounds error
            classificationIter = getSampleRange(sample, classificationSamples, classificationIter)
            trainTimeCol.append(int(sample) * round(int(classificationTimes[classificationIter]) / int(classificationSamples[classificationIter] + 1)))
        patients[patient][0]["Time"] = trainTimeCol
    return patients

def getSampleRange(sample, classificationSamples, classificationIter):
        if(sample > classificationSamples[len(classificationSamples) - 1]):
            classificationIter -= 1
        if(sample > classificationSamples[classificationIter]):
            classificationIter += 1
        return classificationIter


def patientCSVsSparse(patientData, patientLabels, dataOpts, patientNum):
    classificationIter = 0
    classificationTimes = patientLabels["Time"]
    patientCsvRows = []
    tmp = ""
    tmp2 = ""
    cols = patientData.columns
    timeInterval = (patientData.iloc[0]["Time"] + 1)
    shiftedTime = timeInterval
    for index, row in patientData.iterrows():
        if(int(row["Time"]) > int(classificationTimes[classificationIter])):
            if(dataOpts["V#"]):
                patientCsvRows.append(tmp[:-3] + '\n' + tmp2[:-3])
            else:
                patientCsvRows.append(tmp[:-3])
            patientCsvRows.append("")
            classificationIter += 1
            tmp = ""
            tmp2 = ""
            if(int(row["Time"]) >= int(classificationTimes[len(classificationTimes) - 1])):
                break
            if(classificationIter == len(classificationTimes)):
                break
            timeInterval = int(row["Time"])
        if(dataOpts["ST"] == 0):
            shiftedTime = round(row["Time"] - timeInterval) + 1
        tmp = tmp + str(cols[1]) + ',' + str(row[cols[1]]) + ',' + str(int(shiftedTime)) + ' , '
        if(dataOpts["V#"]):
            tmp2 = tmp2 + str(cols[2]) + ',' + str(row[cols[2]]) + ',' + str(int(shiftedTime)) + ' , '
    patientCsvRows = patientCsvRows[:-1] #lazy way to pop last 1 empty lines
    return patientCsvRows, None

def patientCSVsDense(patientData, patientLabels, dataOpts, patientNum):
    classificationIter = 0
    classificationTimes = patientLabels["Time"]
    patientCsvRows = []
    tmp = ""
    tmp2 = ""
    cols = patientData.columns
    for index, row in patientData.iterrows():
        if(int(row["Time"]) > int(classificationTimes[classificationIter])):
            if(dataOpts["V#"]):
                patientCsvRows.append(tmp[:-1] + '\n' + tmp2[:-1])
            else:
                patientCsvRows.append(tmp[:-1])
            classificationIter += 1
            tmp = ""
            tmp2 = ""
            if(int(row["Time"]) >= int(classificationTimes[len(classificationTimes) - 1])):
                break
            if(classificationIter == len(classificationTimes)):
                break
        tmp = tmp + str(row[cols[1]]) + ','
        if(dataOpts["V#"]):
            tmp2 = tmp2 + str(row[cols[2]]) + ','
    return patientCsvRows, patientLabels

def patientCSVsRange(patientData, patientLabels, dataOpts, patientNum):
    patientCsvRows = []
    sampleRange = dataOpts["SR"]
    highestRead = len(patientData["Sample"].tolist())
    cols = patientData.columns
    for index, row in patientLabels.iterrows():
        tmp = ""
        tmp2 = ""
        if(row["Sample"] < sampleRange):
            patientLabels = patientLabels[patientLabels["Sample"] != row["Sample"]]
            continue
        if(row["Sample"] + sampleRange > highestRead):
            patientLabels = patientLabels[patientLabels["Sample"] != row["Sample"]]
            break
        dataRange = patientData[row["Sample"] - sampleRange: row["Sample"] + sampleRange]
        for ind, r in dataRange.iterrows():
            tmp = tmp + str(r[cols[1]]) + ','
            if(dataOpts["V#"]):
                tmp2 = tmp2 + str(r[cols[2]]) + ','
        if(dataOpts["V#"]):
            patientCsvRows.append(tmp[:-1] + '\n' + tmp2[:-1])
        else:
            patientCsvRows.append(tmp[:-1])
    return patientCsvRows, patientLabels

def genPatientCSVs(patientData, patientLabels, dataOpts, patientNum):
    if(dataOpts["SR"] > 0):
        return patientCSVsRange(patientData, patientLabels, dataOpts, patientNum)
    elif(dataOpts["D"] == 1):
        return patientCSVsDense(patientData, patientLabels, dataOpts, patientNum)
    else:
        return patientCSVsSparse(patientData, patientLabels, dataOpts, patientNum)


def classifyFormat(patients, dataOpts):
    for patient in patients.keys():
        patientCsvRows, patientLabels = genPatientCSVs(patients[patient][0], patients[patient][1], dataOpts, patient)
        patients[patient][0] = patientCsvRows
        if(dataOpts["SR"] > 0):
            patients[patient][1] = patientLabels
    return patients

def writeModelData(path, patients, dataOpts):
    with open(f"{path}/classModelData/classData.csv", "w") as gd, open(f"{path}/classModelData/classLabels.csv", "w") as gl:
        for patient in patients.keys():
            with open(f"{path}/features/patient{patient}Data.csv", 'w') as f:
                for line in patients[patient][0]:
                    f.write(f"{line}\n")
                    gd.write(f"{line}\n")
            labels = patients[patient][1]["Type"].tolist()
            with open(f"{path}/labels/patient{patient}Labels.csv", 'w') as f:
                for line in labels:
                    if(dataOpts["B"]):
                        if(int(line) > 0):
                            #dense data -> binary classification
                            gl.write(f"1\n")
                            f.write(f"1\n")
                        else:
                            gl.write(f"0\n")
                            f.write(f"0\n")
                    else:
                        gl.write(f"{int(line)}\n")
                        f.write(f"{int(line)}\n")


if __name__ == "__main__":
    dataOpts = getCmdArgs()
    inputPath = './csv_mitbih_database'
    outputPath = dataOpts['OP']
    patients = readPatientsData(readPatients(inputPath), dataOpts)
    if(dataOpts["SR"]):
        pass
    else:
        patients = moveTimeCol(patients)

    writeModelData(outputPath, removeDeadColumns(classifyFormat(patients, dataOpts)), dataOpts)