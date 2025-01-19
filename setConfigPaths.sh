#run after mvFiles.sh, in automated_testing directory
#moves classify configuration files, and data, to correct locations
#arguments: 1 - dataset to copy over, 2 - timeseries of dataset, 3 - config file 
cp -r ./$1 ../applications/classify/datasets/$2/
cp $3 ../params/classify.json