#Unpack contents of our repo into framework automated testing folder
#   Since we're using some of the same tools, that function in the context of that dir
#   Can edit in this repo without committing changes in framework/managing nested repos
#Argument: relative path from this GH repo to framework automated testing folder
#   Can replace argument with your own fixed/relative path
cp -r ../MIT-BHS-Classify/* ${1}
