import csv
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as mpl

def init_dataframe(path):
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
    annot_path = "./csv_mitbih_database/{}annotations.csv".format(num)
    label_path = "{}/labels/patient{}Labels.csv".format(path, num)
    data_path = "./csv_mitbih_database/{}.csv".format(num)

    df = pd.read_csv(data_path)
    af = pd.read_csv(annot_path)
    lf = pd.read_csv(label_path)
    df = df.drop(['Sample'], axis=1)
    af = af.drop(['Time', 'Type', 'Sub',
                  'Chan', 'Num', 'Aux'], axis=1)

    data = []
    prev_num = 0
    Y = af['Sample'].to_list()
    X = lf['classification'].to_list()
    X.append(0)
    Y.append(649999)
    for l, s in zip(X, Y):
        sample = s
        classification = l
        for i in range(prev_num, sample+1):
            data.append(classification)
        prev_num = sample + 1

    df['classification'] = data
    print(df)   
    return df

def ecg_time(num, path):
    annot_path = "./csv_mitbih_database/{}annotations.csv".format(num)
    label_path = "{}/labels/patient{}Labels.csv".format(path, num)
    data_path = "./csv_mitbih_database/{}.csv".format(num)

    df = pd.read_csv(data_path)
    af = pd.read_csv(annot_path)
    lf = pd.read_csv(label_path)
    df = df.drop(['Sample'], axis=1)
    af = af.drop(['Type', 'Sub',
                  'Chan', 'Num', 'Aux'], axis=1)

    data = []
    data_t = []
    prev_num = 0
    Z = af['time'].to_list()
    Y = af['Sample'].to_list()
    X = lf['classification'].to_list()
    X.append(0)
    Y.append(649999)
    print("Label:{} Annot{}".format(len(X), len(Y)))
    for l, s, t in zip(X, Y, Z):
        time = t
        sample = s
        classification = l
        for i in range(prev_num, sample+1):
            data.append(classification)
            data_t.append(t)
        prev_num = sample + 1

    t_df = pd.DataFrame(columns=['time'], data=data_t)
    l_df = pd.DataFrame(columns=['classifications'], data=data)
    df = pd.concat([df, t_df], axis=1)
    df = pd.concat([df, l_df], axis=1)
    print(df)   
    return df

if __name__ == "__main__":
    label_path = "./csv_mitbih_training/labels"
    #patient_records, num_class = init_dataframe(label_path)
    
    #sb.catplot(data=patient_records, errorbar=None, x="Patient #", y="Instances of Classification", hue="Classification #", kind="bar")
    #mpl.xticks(rotation=70)
    #mpl.show()

    #sb.barplot(data=num_class, errorbar=None, x="Classification #", y="Instances of Classification")
    #mpl.show()

        # Be sure to add 'classification' to the top of the
        # desired patient csv
    #patient_data = read_patient(208, "./dense_csv_mitbih_training")
    #sb.scatterplot(data=patient_data, x="V1", y="MLII", hue="classification")
    #mpl.show()

    dense = ecg_time(208, "./dense_range_pre_pre_SMOTE")
    sb.scatterplot(data=dense, x="time", y="MLII", hue="classification")
    mpl.show()

    binary = ecg_time(208, "./dense_range_pre_SMOTE")
    sb.scatterplot(data=binary, x="time", y="MLII", hue="classification")
    mpl.show()