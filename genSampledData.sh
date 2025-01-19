#want to test quinary and binary. sampled of each
mkdir dense_csv_mitbih_training
mkdir dense_csv_mitbih_training/features
mkdir dense_csv_mitbih_training/labels
mkdir dense_csv_mitbih_training/classModelData
python3 csv_to_training.py --ST 0 --OP dense_csv_mitbih_training --D 1 --V# 1 --N 0 --B 0 --SR 0
cp -r dense_csv_mitbih_training smote_inspired_dense
python3 data_utils.py --denseOversample t 30 1 --OP smote_inspired_dense
cp -r smote_inspired_dense smote_inspired_dense_sampled
python3 data_utils.py --denseUndersample t 30 --OP smote_inspired_dense_sampled

cp -r dense_csv_mitbih_training smote_inspired_dense_binary

python3 data_utils.py --binaryMap smote_inspired_dense_binary
python3 data_utils.py --denseOversample t 30 1 --OP smote_inspired_dense_binary
cp -r smote_inspired_dense_binary smote_inspired_dense_binary_sampled
python3 data_utils.py --denseUndersample t 30 --OP smote_inspired_dense_binary_sampled