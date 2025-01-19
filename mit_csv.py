'''
Assumes presence of empty csv_mitbih_database folder, and mitbih_database folder being in running directory
Creates a CSV for all files in mitbih_database, writes to csv_mitbih_database folder
'''

import os
import csv

def getFiles(path):
    files = []
    for file in os.listdir(path):
        if(file.endswith(".csv")):
            os.system(f"cp ./{path}/{file} ./csv_mitbih_database")
        elif(file.endswith(".txt")):
            files.append(file)
    return files

def extractFileFeatures(path, file):
    lines = None
    with open(path + '/' + file, "r") as f:
        lines = f.readlines()
    for line in range(len(lines)):
        lines[line] = lines[line].split()
        if("#" in lines[line]):
            lines[line].remove("#")
    return lines

def writeCSVs(inputPath, outputPath, files):
    for file in files:
        contents = extractFileFeatures(inputPath, file)
        with open(outputPath + '/' + file[:3] + "annotations.csv", "w") as f:
            write = csv.writer(f)
            write.writerows(contents)


if __name__ == "__main__":
    inputPath = './mitbih_database'
    outputPath = './csv_mitbih_database'
    files = getFiles(inputPath)
    writeCSVs(inputPath, outputPath, files)