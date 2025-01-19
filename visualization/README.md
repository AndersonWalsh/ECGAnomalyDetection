# MIT-BHS-Classify: Visualization

## STEPS TO DO BEFORE RUNNING THE VISUALIZER:
Create the csv_mitbhs_database
 - python3 mit_csv.py
 - python3 csv_val_formatting.py
 
Then do the following commands to create the appropiate folders within the previous directory
 - Create the dense versions of the data
   - python3 csv_to_training.py --ST 0 --OP ./visualization/dense --D 1 --V# 1 --N 0 --B 0
   - copy contents (classModelData, features, and labels) to ./dense_binary
   - python3 data_utils.py --binaryMap ./visualization/dense_binary
 - Create the ranged verson of the data
   - python3 csv_to_training.py --ST 0 --OP ./visualization/range --D 0 --V# 1 --N 0 --B 0 --SR 90
   - copy content to ./ranged_binary
   - python3 data_utils.py --binaryMap ./visualization/ranged_binary
   - copy contents of ./ranged_binary to ./visualization/ranged_SMOTE
   - python3 data_utils.py --SMOTE ./ranged_SMOTE

# Then run python3 vis_patient_features.py
# I will work on adding command args as I will simply produce all visuals.
# As well as the ability to specify what folder to look at.
