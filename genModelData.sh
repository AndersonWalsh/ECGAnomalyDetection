#Automates the entire worflow for generating model data
#   Just runs my python scripts in the right order
#   Can provide commandline args to output directory
rm -r ./csv_mitbih_database/*
python3 mit_csv.py
python3 csv_val_formatting.py
rm -r .$1/*.csv
rm -r .$1/features/*.csv
rm -r .$1/labels/*.csv
rm -r .$1/classModelData/*.csv
python3 csv_to_training.py --OP ./$1