import csv
import sys
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as mpl

def parse_cmd_line():
    cmd = {"Aggregate" : 0, "A_Path" : "../dense_csv_mitbih_training/labels",
            "Read_Pat" : 99, "Data" : "N", "P_Path" : "../dense_csv_mitbih_training", "Num_Feat" : 1, 
            "Binary" : 0, 
            "Prune" : 0, "Max_Len" : 900}
    if len(sys.argv) == 1:
        print("Please run with the command --help")
        exit()
    for i, arg in enumerate(sys.argv):
        try:
            if arg == "--help":
                print("Arguments and Their Defualts:")

                print("--A [PATH TO DATABASE LABELS] - Default: ../dense_csv_mitbih_training/labels")
                print("\t      Aggregation of the classifications from the given path. Reads all label files")
                print("\t      and visualizes patient specific classifications and number of classifications")
                print("\t      as a whole. This is the aggregate() function in the code\n")

                print("--P [#] [N|E|T] [PATH TO PATIENT DATA] [# FEATURES] - Defaults: 100 N ../dense_csv_mitbih_training 1")
                print("NOTE: MANUALLY INSERT THE WORD -> classification A THE TOP OF LABEL CSV")
                print("\t[N]  -> Reads 2 Node Data (MLII and V#) from the csv_mitbih_database and visualizes")
                print("\t        Them against each other and classifies each data point with a classification.")
                print("\t        This is the read_patient() function in the code.\n")

                print("\t[E]  -> Reads all MLII data from patient data file from specified path. This will")
                print("\t        then map the Node data to the time value implicetly set in the formatting")
                print("\t        and visualize as Time vs. MLII with each point colored to its class.")
                print("\t        [NOTE: DATA MUST BE FORMATED IN EITHER DENSE OR RANGED FORMAT AS")
                print("\t         DONE IN csv_to_training.py (--D 1 for dense --SR # for ranged)]\n")

                print("\t[T]  -> Temporal Visualizes the read in MLII vs V# vs Time from the specified path.")
                print("\t        This will make a 3-Dimensional Scatterplot that shows all points in the data.")
                print("\t        [NOTE: DATA SHOULD BE RAN WITH THE --V# 1 FLAG IN csv_to_training.py. ALSO]")
                print("\t         GRAPH IS EXTREAMLY SLOW WHEN REORIENTING AS THERE IS A MASSIVE AMOUNT OF DATA]")

                print("--Q or --B - Default: --Q")
                print("\t      Whether or not the visuals will show Quinary or Binary Classifications")
                print("\t--Q  -> Shows all classifications (0-4) as specified by Philip de Chazal; M. O'Dwyer; and R.B. Reilly")
                print("\t        in 'Automatic classification of heartbeats using ECG morphology and heartbeat interval features'")
                print("\t--B  -> Shows classifications as a binary Normal (0) and Abnormal (1-4)")

                print("--Pr [MAX LENGTH OF ROW TO VISUALIZE] - Default: 900")
                print("\t      ")
                exit()
            match arg:
                case "--A":
                    cmd["Aggregate"] = 1
                    try:
                        cmd["A_Path"] = sys.argv[i + 1]
                    except:
                        print("Error or path not provided, assuming default for Aggregate Path: ../dense_csv_mitbih_training/labels")
                        continue
                case "--P":
                    j = 1
                    try:
                        cmd["Read_Pat"] = int(sys.argv[i + j])
                        j += 1
                    except:
                        print("Error or path not provided, assuming default for Patient Number: 100")
                    try:
                        cmd["Data"] = sys.argv[i + j]
                        j += 1
                    except:
                        print("Error or path not provided, assuming default for Patient Data Type: N")
                    try: 
                        cmd["P_Path"] = sys.argv[i + j]
                        j += 1
                    except:
                        print("Error or path not provided, assuming default for Pathient Data Path: ../dense_csv_mitbih_training")
                    try: 
                        cmd["Num_Feat"] = int(sys.argv[i + j])
                        j += 1
                    except:
                        print("Error, assuming default for Number of Features: 1")
                        continue
                case "--B":
                    cmd["Binary"] = 1
                case "--Pr":
                    cmd["Prune"] = 1
                    try:
                        cmd["Max_Len"] = int(sys.argv[i + 1])
                    except:
                        print("Error or path not provided, assuming default of Max Row Length: 900")
        except:
            print("Error in the command line. Please run this program with only --help")
            exit()

    return cmd

def aggregate(path):
    df = pd.DataFrame()
    total_class = [0, 0, 0, 0, 0]
    
    for num in range(100, 234+1):
        num_0 = num_1 = num_2 = num_3 = num_4 = 0
        file = path + "/patient{}Labels.csv".format(num)
        try:
            with open(file, "r") as patient_csv:
                reader = csv.reader(patient_csv)
                for line in reader:
                    if   int(line[0]) == 0:
                        num_0 += 1
                        total_class[0] += 1
                    elif int(line[0]) == 1:
                        num_1 += 1
                        total_class[1] += 1
                    elif int(line[0]) == 2:
                        num_2 += 1
                        total_class[2] += 1
                    elif int(line[0]) == 3:
                        num_3 += 1
                        total_class[3] += 1
                    elif int(line[0]) == 4:
                        num_4 += 1
                        total_class[4] += 1

            tmp_df = pd.DataFrame({"Patient #" : [str(num)], "Classification 0" : [num_0], "Classification 1" : [num_1],
                                    "Classification 2" : [num_2], "Classification 3" : [num_3], "Classification 4" : [num_4]})
            
            df = pd.concat([df, tmp_df])
        except:
            continue
    tc_df = pd.DataFrame({"Classification 0" : [total_class[0]], "Classification 1" : [total_class[1]], 
                          "Classification 2" : [total_class[2]], "Classification 3" : [total_class[3]], 
                          "Classification 4" : [total_class[4]]})
    
    df = pd.melt(df, id_vars="Patient #", var_name="Classification #", value_name="Instances of Classification")
    tc_df = pd.melt(tc_df, var_name="Classification #", value_name="Instances of Classification")
    return df, tc_df

def read_patient(num, path):
    annot_path = "../csv_mitbih_database/{}annotations.csv".format(num)
    label_path = "{}/labels/patient{}Labels.csv".format(path, num)
    data_path = "../csv_mitbih_database/{}.csv".format(num)

    df = pd.read_csv(data_path)
    af = pd.read_csv(annot_path)
    try:
        lf = pd.read_csv(label_path)
    except:
        print("Make sure you put the word classification at the top of the lable file located in:\n{}".format(label_path))
        exit()
    df = df.drop(['Sample'], axis=1)
    af = af.drop(['Time', 'Type', 'Sub',
                  'Chan', 'Num', 'Aux'], axis=1)

    data = []
    prev_num = 0
    Y = af['Sample'].to_list()
    X = lf['classification'].to_list() # MAKE SURE 'classification' IS
    X.append(0)                        # AT THE TOP OF THE LABEL CSV
    Y.append(649999)
    for l, s in zip(X, Y):
        sample = s
        classification = l
        for i in range(prev_num, sample+1):
            data.append(classification)
        prev_num = sample + 1

    df['classification'] = data
    return df

def ecg_time_dense(num, path, num_feat):
    annot_path = "../csv_mitbih_database/{}annotations.csv".format(num)
    label_path = "{}/labels/patient{}Labels.csv".format(path, num)
    data_path = "{}/features/patient{}Data.csv".format(path, num)

    af = pd.read_csv(annot_path)
    lf = pd.read_csv(label_path)
    af = af.drop(['Type', 'Sub',
                  'Chan', 'Num', 'Aux'], axis=1)

    X = []  # MLII
    Y = []  # V#
    H = lf['classification'].to_list() # MAKE SURE 'classification' IS 
                                       # AT THE TOP OF THE LABEL CSV

    '''
        Each line has X amount of Observations
        Each "Column" in each line has an
            implicet time based on the
            column number
        Each line has a corrisponding
            classification
    '''
    with open(data_path, "r") as data_csv:
        reader = csv.reader(data_csv)

        row_num = 0 # Odd -> MLII, Even -> V# 
        for line in reader:
            data_x = []
            data_y = []
            for ecg in line:
                if row_num % num_feat == 0:
                    data_x.append(float(ecg))  # MLII
                else:
                    data_y.append(float(ecg))
            #print("Row number: {}".format(row_num))
            if row_num % num_feat == 0:
                #print("\t\tMLII: {}".format(len(data_x)))
                X.append(data_x)  # MLII
            else:
                #print("\t\tV#: {}".format(len(data_y)))
                Y.append(data_y)
            row_num += 1

    return X, Y, H

if __name__ == "__main__":
    cmd = parse_cmd_line()
    '''
        Show all the classification data
    '''
    if cmd["Aggregate"]:
        patient_records, num_class = aggregate(cmd["A_Path"])
        
        sb.catplot(data=patient_records, errorbar=None, x="Patient #", y="Instances of Classification", hue="Classification #", kind="bar")
        mpl.xticks(rotation=70)
        mpl.show()

        sb.barplot(data=num_class, errorbar=None, x="Classification #", y="Instances of Classification")
        mpl.xticks(rotation=70)
        mpl.show()

    '''
        Patient Specific Data
            Make sure to add 'classification' to the top
                of the Label file you are visualizing
                    To Do: Figure out how to do this in
                    the program itself so it does not
                    have to be done manually.
    '''
    if cmd["Read_Pat"] >= 100:
        if cmd["Data"]   == "N" or cmd["Data"] == "[N]":
            patient_data = read_patient(cmd["Read_Pat"], cmd["P_Path"])
            V = patient_data.columns[1]
            sb.scatterplot(data=patient_data, x="0", y="{}".format(V), hue="classification", palette="flare")
            mpl.show()
        else:
            X, Y, H = ecg_time_dense(cmd["Read_Pat"], cmd["P_Path"], cmd["Num_Feat"])
            color = {0 : "#440154", 1 : "#21918b", 2 : "#5ec961", 3 : "#fbeb58", 4 : "#000000"}

            if   cmd["Prune"]:
                for i, row in enumerate(X):
                    if len(row) > cmd["Max_Len"]:
                        n_mlii = np.resize(row, cmd["Max_Len"])
                        n_v = np.resize(Y[i], cmd["Max_Len"])
                        X[i] = n_mlii.tolist()
                        Y[i] = n_v.tolist()

            if   cmd["Data"] == "E" or cmd["Data"] == "[E]":
                for i, row in enumerate(X):
                    time = [x for x in range(len(row))]
                    sb.scatterplot(x=time, y=row, color=color[H[i]])
                mpl.xlabel("Time")
                mpl.ylabel("MLII#")
                mpl.legend()
                mpl.show()
            elif cmd["Data"] == "T" or cmd["Data"] == "[T]":
                if cmd["Num_Feat"] < 2:
                    print("Number of specified features must be 2 for temporal visualization")
                    exit()

                fig = mpl.figure()
                axes = mpl.axes(projection='3d')
                num_skipped = 0
                for i, row in enumerate(X):
                    time = [x for x in range(len(row))]
                    if len(row) != len(Y[i]):
                        print("Row Number: {}\n\tX/MLII:{}\n\tY/V#:{} \n\tDifference: {}\n".format(i, len(row), len(Y[i]), len(row) - len(Y[i])))
                        with open("{}_error.txt".format(cmd["Read_Pat"]), "a") as err:
                            err.write("Row Number: {}\n\tX/MLII:{}\n\tY/V#:{} \n\tDifference: {}\n".format(i, len(row), len(Y[i]), len(row) - len(Y[i])))
                            err.write("{}\n{}\n\n".format(row, Y[i]))
                        num_skipped += 1
                        continue
                    axes.scatter(xs=time, ys=Y[i], zs=row, color=color[H[i]])
                axes.set_zlabel("MLII")
                axes.set_xlabel("Time")
                axes.set_ylabel("V#")
                print("Skipped due to formating error: {}".format(num_skipped))
                mpl.show()

                if cmd["Binary"]:
                    fig_b = mpl.figure()
                    axes_b = mpl.axes(projection='3d')
                    for j, row_b in enumerate(X):
                        binary = 0
                        if H[j] > 0:
                            binary = 1
                        time_b = [x for x in range(len(row_b))]
                        axes_b.scatter(xs=time_b, ys=Y[j], zs=row_b, color=color[binary])
                    axes_b.set_zlabel("MLII")
                    axes_b.set_xlabel("Time")
                    axes_b.set_ylabel("V#")
                    mpl.show()