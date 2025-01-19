'''
NOTE: THIS VERSION EXISTS SOLELY FOR ego_opt_search.json USAGE. Other cases may break
Author: Anderson Walsh
Conducts training for every permutation of training. Permutations produced from values in user specified JSON file
    - Stores real-time commandline output metrics in txt files corresponding to test number (call that generated the file corresponds to test<num - 1>.txt = Run <num>), in network_results 
    - Call that generated test file also stored at top of file
    - Stores generated networks in generated_networks, named <app>_<proc><test num>
    - Integers corresponding to tests shown at commandline
Args:
    - JSON file containing specifications for commandline
        - python3 eons_runs_bio.py <your json>.json

General workflow:
"
#For each permutation, moves to cpp-apps
make app=<app_name> proc=<neuroprocessor>
#One possible list of arguments, not limited to this
<path to compiled app> -a train --threads <num threads> --episodes <num episodes> --extra_eons_params <eons params, json> --extra_proc_params {processor params, json} --encoder <encoder params, json> --max_fitness <num> --epochs <integer> --seed <int> <extra user specified args>
"

You can add your own parameters for testing
    - Expected to be of format "<new command line arg>": [<1 or more arguments>]
        Example: append to file ', "initial_min_fitness": [<comma separated nums>]' no single quotes
        - for args that have their own args (extra_eons_params), supply json object with values to permutate within as lists (ie, "encoder": {{"max_spikes": [4,6]}})

Notes: 
    - can adjust default parameters of network/application in params directory
    - "extra_params_static" will not result in new permutations
        - recommended to only use params that do not affect console output
        - expected that lists within only contain 1 element
    - episodes exist within epochs
        - average across all network seeds given as fitness
    - For params that are json themselves (extra_eons_params, extra_proc_params, etc), also format as json objects

TODO: 
    - Documentation
    - Validation of behavior across more varied JSON
    - Currently assumes default relative paths. Allow user to specify
    - Add option to specify which runs to execute with greater specificity
    - More robust support for permutations of coding schemes
BUG FIX:
    - Account for empty fields outside of static params
    - Nested JSON objects break the code in some cases
    - Specifying coding schemes can break the script
'''

import sys
import os
import json
import itertools

#stuff specific to bio, should fully generalize later
'''
Change target data for each tested permutation
'''
def folderNames():
    pass

def getTrainingParams():
    filename = None
    try:
        filename = sys.argv[1]
    except IndexError as err:
        print("Provide commandline argument: <path to eons run paramater json file>")
        exit()
    if(not os.path.isfile(filename)):
        print("Argument is not a file")
        exit()
    args = None
    with open(filename, "r") as params:
        args = json.load(params)
    return args


def genNetworkPermutations(params):
    #for param in param
    #explicitly start list with app name
    paramList = [params['apps']]
    params.pop("apps")
    #start extracting commandline keywords, separate values
    paramNames = []
    #extract static parameters, no permutations
    unpackStaticParams(params, paramNames, paramList)
    for param in params.keys():
        if(type(params[param]) is dict):
            #for json params, to allow usage of itertools.permutate, unpack values and denote number of values in paramNames
            recurUnpackJson(paramNames, paramList, params, param)
        else:
            paramNames.append(f"--{param}")
            paramList.append(params[param])
    permutations = list(itertools.product(*paramList))
    return paramNames, permutations
    

def recurUnpackJson(paramNames, paramList, params, param):
    try:
        paramNames.append([param, len(params[param].values())])
    except AttributeError as err:
            paramList.append(params[param])
            paramNames.append(param)
            return
    for jsonParam in params[param]:
        if(type(params[param]) is dict):
            recurUnpackJson(paramNames, paramList, params[param], jsonParam)
        else:
            paramList.append(params[param][jsonParam])
            paramNames.append(jsonParam)


def unpackStaticParams(params, paramNames, paramList):
    for static_param in params['extra_params_static'].keys():
        #no empty lists
        if(params['extra_params_static'][static_param] == []):
            continue
        paramList.append(params['extra_params_static'][static_param])
        paramNames.append(static_param)
    params.pop("extra_params_static")

def incList(paramNames, k, listCounter=1):
    if(type(paramNames[k]) is list and paramNames[k][1] != 0):
        listCounter += 1
        k += paramNames[k][1]
        k, listCounter = incList(paramNames, k, listCounter)
    return k, listCounter
    

def cmdCalls(paramNames, permutations):
    cmdCalls = []
    for i in range(len(permutations)):
        cmdStr = f"{permutations[i][0]} -a train "
        k = 0
        decVals = 0
        while k < len(paramNames):
            if(type(paramNames[k]) is list and paramNames[k][1] > 0):
                cmdStr += buildCmdJson(paramNames, permutations[i], k, base=True)
                cmdStr += "'"
                tmp = incList(paramNames, k)
                k = tmp[0]
                decVals = tmp[1]
            elif(paramNames[k][1] == 0):
                paramNames[k][0] = None #this case is producing an intermittent bug
            else:
                cmdStr += f" --{paramNames[k]} "
                cmdStr += f"{permutations[i][k + 1 - decVals]}"
            k += 1
        cmdCalls.append(cmdStr)
    return cmdCalls
    

def buildCmdJson(paramNames, permutations, curInd, cmdStr="", base=False, permInd = 0):
    #in param names str, number of json elements belonging to that param name are stored as immediate next index in a list with it
    if(base):
        cmdStr += f" --{paramNames[curInd][0]}"
        cmdStr += " '{"
    else:
        cmdStr=""
    for i in range(1, paramNames[curInd][1]+1):
        if(type(paramNames[curInd + i]) is list):
            cmdStr += f'"{paramNames[curInd + i][0]}": '
            cmdStr += "{"
            cmdStr += buildCmdJson(paramNames, permutations, curInd + i, cmdStr)
        else:
            cmdStr += f'"{paramNames[curInd + i]}": '
            k = 1
            while(k < len(permutations)):
                try:
                    cmdStr += f"{permutations[curInd + i - k]}"
                    break
                except IndexError as err:
                    k += 1
                if(k == len(permutations)):
                    print("There was an error in processing the input JSON. Exiting")
                    exit()
                
            if(i + 1 < paramNames[curInd][1]+1):
                cmdStr += ", "
            else:
                cmdStr += '}'
                try:
                    if(type(paramNames[curInd-1]) is list):
                        cmdStr += '}'
                except:
                    print("Error checking for end of JSON object")
                    exit()
    return cmdStr

def makeApps(params, makeAppsFlag=True):
    os.chdir("../")
    compiled_apps = []
    for app in params['app']:
        for proc in params['proc']:
            if(makeAppsFlag):
                os.system(f"make app={app} proc={proc}")
            compiled_apps.append(f"bin/{app}_{proc}")
    os.chdir("./automated_testing/")
    return compiled_apps

def patchDecoder(prompts, decodeStyle):
    for i in range(len(prompts)):
        ind = prompts[i].rfind("}}")
        tmp = prompts[i][:ind]
        ind2 = tmp.rfind("50")
        if(decodeStyle == "WTA"):
            prompts[i] = prompts[i][:ind2] + "\"wta\"" + prompts[i][ind:]
        else:
            prompts[i] = prompts[i][:ind2] + "true" + prompts[i][ind:]
    return prompts

def runCommands(prompts, startInd=0):
    os.chdir("../")
    if(startInd < 0):
        print("Are you just trying to break stuff?")
        exit()
    if(startInd != 0):
        startInd -= 1
    testNum = startInd
    for i in range(startInd, len(prompts)):
        print(f"Running {prompts[i]}")
        echoStr = ""
        for k in range(len(prompts[i])):
            if(prompts[i][k] == '"' or prompts[i][k] == '"'):
                echoStr += "\\"
            echoStr += prompts[i][k]
        #for folder in folders
        #   for file in data files
        #       update path pointed to by classify.json
        for folder in ["./applications/classify/datasets/dense/log_dense_mlii_binary", "./applications/classify/datasets/dense/zscore_dense_mlii_binary", "./applications/classify/datasets/dense/log_ranged_binary", "./applications/classify/datasets/dense/, zscore_ranged_binary"]:#["./applications/classify/datasets/dense/dense_mlii_mitbih_binary", "./applications/classify/datasets/dense/dense_ranged_mlii_mitbih_binary", "./applications/classify/datasets/dense/dense_ranged_mlii_mitbih_binary_oversampled"]:# ecg opt search 1 applied here, current is ecg opt search 3["./applications/classify/datasets/dense/dense_ranged_mitbih", "./applications/classify/datasets/dense/dense_ranged_mitbih_normalized", "./applications/classify/datasets/dense/dense_ranged_mitbih_binary_normalized"]:
            for file in os.listdir(f"{folder}/features"):
                if("Data.csv" in file):
                    classify_params = None
                    with open("./params/classify.json", 'r') as f:
                        classify_params = json.load(f)
                        classify_params["data_csv"] = f"{folder}/features/{file}"
                        classify_params["label_csv"] = f"{folder}/labels/{file[:-8]}Labels.csv"
                    with open("./params/classify.json", 'w') as f:
                        json.dump(classify_params, f)

                    filePrepend = folder[folder.rfind("dense/")+6:]
                    os.system(f'echo "{echoStr}" > ./automated_testing/network_results/{filePrepend}test{testNum}patient{file[-11:-8]}.txt')
                    os.system(f"{prompts[i]} -t dense >> ./automated_testing/network_results/{filePrepend}test{testNum}patient{file[-11:-8]}.txt") #part of the laziest 1am fixes of all time
                    networkName = prompts[i].split()[0][4:]
                    os.system(f"cp ./networks/{networkName}_train.txt ./automated_testing/generated_networks/{filePrepend}{networkName}{testNum}patient{file[-11:-8]}.txt")


        testNum += 1
    os.chdir("./automated_testing/")

if(__name__ == "__main__"):
    params = getTrainingParams()
    makeAppsFlag = True
    decodeStyle = "WTA"
    try:
        makeOpt = sys.argv[2]
        if(makeOpt == '--false'):
            makeAppsFlag = False
            if(not makeAppsFlag):
                decodeStyle = "temp"
    except IndexError:
        print("Compiling apps")
    compiled_apps = makeApps(params, True) 
    params.pop("app")
    params.pop("proc")
    params.update({"apps": compiled_apps})
    paramNames, permutations = genNetworkPermutations(params)
    prompts = cmdCalls(paramNames, permutations)
    prompts = patchDecoder(prompts, decodeStyle) 
    for i in range(len(prompts)):
        print(f"Run {i+1}: {prompts[i]}")
    runCommands(prompts, startInd=1)
    '''try:
        userTest = int(input("Recommended to scan the queued tests and validate they appear as you expect. Run these tests? (Pass positive int for run to start from (1 to go from beginning), or 0 to exit): "))
        if(userTest):
            print(userTest)
            #runCommands(prompts, startInd=userTest)
        else:
            print("Exiting")
    except ValueError:
        print("Requires integer >= 1 or 0")
    '''