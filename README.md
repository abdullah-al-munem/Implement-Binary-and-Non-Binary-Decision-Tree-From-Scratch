# Implement-Binary-and-Non-Binary-Decision-Tree-From-Scratch

Parameters/Switches: 

1. -m measureName (measureName  can be "G" (for gini index) or "E" (for entropy). If no "-a" switch is found, default will be Gini Index)
2. -f fileName (fileName is the name of the dataset. By default the dataset is "Irish" which is available in a certain package of python and on internet)
3. -br branching-factor (branching-factor = the branching factor to decide whether the tree is binary or non-binary. The default value is "b" for binary Decision Tree and "nb" for non-binary tree)
4. -t testFileName (testFileName is the name of a file name that contains testing data, at least a single row should be there. If "-t" is specified then you need to use the tuples in the file and display the class labels for the testing tuples). 
5. -r ratio (ratio is a floating point value where 0.7 means 70% of the data will be used for training and the remaining for testing. If this number is not specified then 100% data will be used for training. The testing data will be used exactly as "-t" switch is handled. Moreover, any command with -t and -r switches together should be warned and one of them has to be ignored.)
6. -c classLabel (classLabel represent the name of the attribute of the dataset that is going to be used as target attribute. In case of this parameter is missing, use the last attribute of the dataset as target class label.)
7. -d display-node-order (display-node-order shows the order of the nodes in the tree in prefix style [Root, Left, Right] that reveals the order of the splits. This switch have no parameters to take as input.)


Sample Inputs: 
1. python DTC.py -r 0.6 -d
2. python DTC.py -f creditcard.csv -r 0.8 -d -m G
3. python DTC.py -h strange.csv 0.8 -m G
4. python DTC.py -f Iris.csv -r 0.7 -d -m E -br b
5. python DTC.py -f creditcard.csv -r 0.7 -d -m E -br nb
6. python DTC.py -f creditcard.csv -r 0.7 -d -m E -br nb
7. python DTC.py -f Iris.csv -r 0.7 -d -m E -br cb
8. python DTC.py -f Iris.csv -r 0.7 -d -m E -br cb
9. python DTC.py -f creditcard.csv -r 0.7 -d -m E -br
10. python DTC.py -f creditcard.csv -r 0.7 -d -m F -br b
11. python DTC.py -f Iris.csv -r 0.7 -d -m E -br nb
12. python DTC.py -f Iris.csv -r 0.7 -d -m E -br nb
