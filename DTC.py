import os
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import BinaryTree

# This function is for encoding string/object data type columns
def Encode_Label(df):
    dtype = df.dtypes
    idx = np.array(dtype.index)
    le = LabelEncoder()
    for i in idx:
        if df[i].dtypes == CategoricalColumn.dtypes:
            df[i] = le.fit_transform(df[i])

    return

# For creating a binary tree from dictionary 
def make_binary_tree(tree, node):
    '''if not isinstance(tree, dict):
        node = BinaryTree.Node(tree)
        return node'''

    question = list(tree.keys())[0]
    node = BinaryTree.Node(question)

    if not isinstance(tree[question][0], dict):
        node.left = BinaryTree.Node(tree[question][0])
    else:
        residual_tree = tree[question][0]
        node.left = make_binary_tree(residual_tree, node.left)

    if not isinstance(tree[question][1], dict):
        node.right = BinaryTree.Node(tree[question][1])
    else:
        residual_tree = tree[question][1]
        node.right = make_binary_tree(residual_tree, node.right)

    return node

# Get the column index of the class label
def get_ClassLabel(df, ClassLabel):
    colum_num = -1
    j = 0
    for i in df.columns:
        if i == ClassLabel:
            colum_num = j
            break
        j += 1
    return colum_num

# For checking a dataset or subdataset is pure or not. 
# Pure means there is only one class label. 
def check_purity(data, ClassLabel):
    label_column = data[:, ClassLabel]
    unique_classes = np.unique(label_column)

    if len(unique_classes) == 1:
        return True
    else:
        return False

# Classifying the dataset
def classify_data(data, ClassLabel):
    label_column = data[:, ClassLabel]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)

    index = counts_unique_classes.argmax()
    classification = unique_classes[index]

    return classification

# Get potential splits for every single unique value
def get_potential_splits(data, ClassLabel, max_split=2):
    potential_splits = {}
    n_columns = data.shape
    n_columns = n_columns[1]
    for column_index in range(n_columns):
        if column_index == ClassLabel:
            continue
        values = data[:, column_index]
        unique_values = np.unique(values)

        potential_splits[column_index] = unique_values

    return potential_splits

# Get potential splits according to the  max split criteria
# (This function will must include the last unique value)
def get_potential_splits2(data, ClassLabel, max_split=2):
    potential_splits = {}
    n_columns = data.shape
    n_columns = n_columns[1]
    for column_index in range(n_columns):
        if column_index == ClassLabel:
            continue
        values = data[:, column_index]
        unique_values = np.sort(np.unique(values))
        type_of_feature = FEATURE_TYPES[column_index]
        if len(unique_values) > max_split and type_of_feature == "continuous":
            potential_splits[column_index] = []
            idx = (len(unique_values)//max_split)
            for i in range(1, idx+1):
                if idx*i <= len(unique_values):
                    potential_splits[column_index].append(unique_values[(idx*i)-1])
            if len(potential_splits[column_index]) < max_split:
                limit = max_split - len(potential_splits[column_index])
                for i in range(1,limit+1):
                    if unique_values[-i] in potential_splits[column_index]:
                        i -= 1
                        continue
                    potential_splits[column_index].append(unique_values[-i])
            else:
                limit = len(potential_splits[column_index]) - max_split
                for i in range(1, limit + 2):
                    potential_splits[column_index].pop(-i)
                potential_splits[column_index].append(unique_values[-1])

        else:
            potential_splits[column_index] = unique_values

    return potential_splits

# Get potential splits according to the  max split criteria
# The last unique value may or may not be include
def get_potential_splits3(data, ClassLabel, max_split=2):
    potential_splits = {}
    n_columns = data.shape
    n_columns = n_columns[1]
    for column_index in range(n_columns):
        if column_index == ClassLabel:
            continue
        values = data[:, column_index]
        unique_values = np.sort(np.unique(values))

        type_of_feature = FEATURE_TYPES[column_index]
        if len(unique_values) > max_split and type_of_feature == "continuous":
            potential_splits[column_index] = []
            idx = len(unique_values)//max_split
            for i in range(1, idx+1):
                if idx*i <= len(unique_values):
                    potential_splits[column_index].append(unique_values[(idx*i)-1])
        else:
            potential_splits[column_index] = unique_values

    return potential_splits

# Splitting the dataset based on potential split value
# This function is only for binary splits (Binary DT)
def split_data(data, split_column, split_value):
    split_column_values = data[:, split_column]

    type_of_feature = FEATURE_TYPES[split_column]
    if type_of_feature == "continuous":
        data_below = data[split_column_values <= split_value]
        data_above = data[split_column_values > split_value]

    else:
        data_below = data[split_column_values == split_value]
        data_above = data[split_column_values != split_value]

    return data_below, data_above

# This function is for non binary splits (Non Binary DT)
# Used for categorical features
def split_data2(data, split_column, split_value):
    split_column_values = data[:, split_column]

    data_equal= data[split_column_values == split_value]
    return data_equal

# Used for continuous features
def split_data3(data, split_column, split_value, pre_value):
    split_column_values = data[:, split_column]

    data_equal= data[(split_column_values <= split_value) & (split_column_values > pre_value)]
    return data_equal

# Calculate the entropy of the dataset
def calculate_entropy(data, ClassLabel):
    label_column = data[:, ClassLabel]
    counts = np.unique(label_column, return_counts=True)
    counts = counts[1]

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))

    return entropy

# This function used in determine_best_split() which is used for binary DT
def calculate_overall_entropy(data_below, data_above, ClassLabel):
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    overall_entropy = (p_data_below * calculate_entropy(data_below, ClassLabel)
                       + p_data_above * calculate_entropy(data_above, ClassLabel))

    return overall_entropy

# This function used in determine_best_split2() which is used for non binary DT
def calculate_overall_entropy2(data, list_n, entropy):
    n = len(data)

    overall_entropy = 0
    for i in range(len(entropy)):
        p_data = list_n[i] / n
        overall_entropy += (p_data * entropy[i])

    return overall_entropy

# Calculate the gini of the dataset
def calculate_gini(data, ClassLabel):
    label_column = data[:, ClassLabel]
    counts = np.unique(label_column, return_counts=True)
    counts = counts[1]

    probabilities = counts / counts.sum()
    gini = 1 - sum(probabilities**2)

    return gini

# This function used in determine_best_split() which is used for binary DT
def calculate_overall_gini(data_below, data_above, ClassLabel):
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    overall_gini = (p_data_below * calculate_gini(data_below, ClassLabel)
                       + p_data_above * calculate_gini(data_above, ClassLabel))

    return overall_gini

# This function used in determine_best_split2() which is used for non binary DT
def calculate_overall_gini2(data, list_n, gini):
    n = len(data)

    overall_gini = 0
    for i in range(len(gini)):
        p_data = list_n[i] / n
        overall_gini += (p_data*gini[i])

    return overall_gini

# This function will determine the best split for the given dataset
# (Only Binary Split)
def determine_best_split(data, potential_splits, ClassLabel, MeasureName, DONE=[]):

    overall_entropy = 9999
    overall_gini = 9999
    for column_index in potential_splits:
        if COLUMN_HEADERS[column_index] in DONE:
            continue
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            if MeasureName == 'E':
                current_overall_entropy = calculate_overall_entropy(data_below, data_above, ClassLabel)
                if current_overall_entropy <= overall_entropy:
                    overall_entropy = current_overall_entropy
                    best_split_column = column_index
                    best_split_value = value
            else:
                current_overall_gini = calculate_overall_gini(data_below, data_above, ClassLabel)
                if current_overall_gini <= overall_gini:
                    overall_gini = current_overall_gini
                    best_split_column = column_index
                    best_split_value = value



    return best_split_column, best_split_value

# This function will determine the best split for the given dataset
# (Non Binary Split but can be binary if max_split == 2)
def determine_best_split2(data, potential_splits, ClassLabel, MeasureName, DONE=[]):
    overall_entropy = 9999
    overall_gini = 9999
    entropy = []
    gini = []
    list_n = []
    '''best_split_column = None
    best_split_value = None'''
    for column_index in potential_splits:
        if COLUMN_HEADERS[column_index] in DONE:
            continue
        n = len(potential_splits[column_index])
        type_of_feature = FEATURE_TYPES[column_index]

        if type_of_feature == "continuous":
            pre_value = 0
            for value in potential_splits[column_index]:

                data_equal = split_data3(data, split_column=column_index, split_value=value, pre_value=pre_value)
                list_n.append(len(data_equal))
                if MeasureName == 'E':
                    entropy.append(calculate_entropy(data_equal, ClassLabel))
                else:
                    gini.append(calculate_entropy(data_equal, ClassLabel))
                pre_value = value
            if MeasureName == 'E':
                current_overall_entropy = calculate_overall_entropy2(data, list_n, entropy)
                if current_overall_entropy <= overall_entropy:
                    overall_entropy = current_overall_entropy
                    best_split_column = column_index
                    best_split_value = value
            else:
                current_overall_gini = calculate_overall_gini2(data, list_n, gini)
                if current_overall_gini <= overall_gini:
                    overall_gini = current_overall_gini
                    best_split_column = column_index
                    best_split_value = value

        else:
            for value in potential_splits[column_index]:
                data_equal = split_data2(data, split_column=column_index, split_value=value)
                list_n.append(len(data_equal))
                if MeasureName == 'E':
                    entropy.append(calculate_entropy(data_equal, ClassLabel))
                else:
                    gini.append(calculate_entropy(data_equal, ClassLabel))

            if MeasureName == 'E':
                current_overall_entropy = calculate_overall_entropy2(data, list_n, entropy)
                if current_overall_entropy <= overall_entropy:
                    overall_entropy = current_overall_entropy
                    best_split_column = column_index
                    best_split_value = value
            else:
                current_overall_gini = calculate_overall_gini2(data, list_n, gini)
                if current_overall_gini <= overall_gini:
                    overall_gini = current_overall_gini
                    best_split_column = column_index
                    best_split_value = value

    return best_split_column, best_split_value

# Detect the type of attributes
def determine_type_of_feature(df, ClassLabel):
    feature_types = []
    for feature in df.columns:
        if feature != ClassLabel:
            unique_values = df[feature].unique()
            example_value = unique_values[0]

            if isinstance(example_value, str):
                feature_types.append("categorical")
            else:
                feature_types.append("continuous")
    return feature_types

# For finding the column index of a specific column
def get_column_number2(df, column):
    j=0
    for i in df.columns:
        if i == column:
            idx = j
        j += 1
    return idx

# Get the column index for every single column
def get_column_number(df):
    column_number = {}
    j=0
    for i in df.columns:
        column_number.update({i : j})
        j += 1
    return column_number

# The Decision tree algorithm
'''

For understanding the parameters, please read the "Instructions.txt" file. 

'''
def decision_tree_algorithm(df, ClassLabel, MeasureName, BFactor, max_split=2, counter=0, max_depth=10, DONE = [], flg=0):
    if counter == 0:
        global COLUMN_HEADERS, FEATURE_TYPES, DONE2, COLUMN_NUMBER
        DONE2 = []
        COLUMN_HEADERS = df.columns
        FEATURE_TYPES = determine_type_of_feature(df, ClassLabel)
        COLUMN_NUMBER = get_column_number(df)
        data = df.values
        #ClassLabelx = ClassLabel
        ClassLabel = get_ClassLabel(df, ClassLabel)

    else:
        data = df

    if (check_purity(data, ClassLabel)) or (counter == max_depth):
        classification = classify_data(data, ClassLabel)

        return classification

    else:
        counter += 1
        if BFactor == 'b':
            #potential_splits = get_potential_splits(data, max_split)
            if flg==0:
                potential_splits = get_potential_splits2(data, ClassLabel, max_split)
            elif flg == 1:
                potential_splits = get_potential_splits3(data, ClassLabel, max_split)
            elif flg == 2:
                potential_splits = get_potential_splits(data, ClassLabel, max_split)

            split_column, split_value = determine_best_split(data, potential_splits, ClassLabel, MeasureName, DONE)
            data_below, data_above = split_data(data, split_column, split_value)

            if len(data_below) == 0 or len(data_above) == 0:
                classification = classify_data(data, ClassLabel)
                return classification

            feature_name = COLUMN_HEADERS[split_column]
            type_of_feature = FEATURE_TYPES[split_column]

            if type_of_feature == "continuous":
                question = "{} <= {}".format(feature_name, split_value)

            else:
                question = "{} = {}".format(feature_name, split_value)

            sub_tree = {question: []}

            yes_answer = decision_tree_algorithm(data_below, ClassLabel, MeasureName, BFactor, max_split, counter, max_depth, DONE, flg)
            no_answer = decision_tree_algorithm(data_above, ClassLabel, MeasureName, BFactor, max_split, counter, max_depth, DONE, flg)

            if yes_answer == no_answer:
                sub_tree = yes_answer
            else:
                sub_tree[question].append(yes_answer)
                sub_tree[question].append(no_answer)

            return sub_tree
        else:
            if flg==0:
                potential_splits = get_potential_splits2(data, ClassLabel, max_split)
            elif flg == 1:
                potential_splits = get_potential_splits3(data, ClassLabel, max_split)
            elif flg == 2:
                potential_splits = get_potential_splits(data, ClassLabel, max_split)

            if len(DONE) == (len(COLUMN_NUMBER)-1):
                DONE.clear()

            split_column, split_value = determine_best_split2(data, potential_splits, ClassLabel, MeasureName, DONE)
            '''if split_column == None or split_value == None:
                return'''

            feature_name = COLUMN_HEADERS[split_column]
            type_of_feature = FEATURE_TYPES[split_column]

            DONE.append(feature_name)
            node = "{}".format(feature_name)
            sub_tree2 = {node: []}
            pre_value = 0
            for value in potential_splits[split_column]:
                if type_of_feature == "continuous":
                    data_below = split_data3(data, split_column, value, pre_value)

                    if len(data_below) == 0:
                        classification = classify_data(data, ClassLabel)
                        return classification

                    question = "{} <= {}".format(feature_name, value)
                    sub_tree = {question: []}

                    answer = decision_tree_algorithm(data_below, ClassLabel, MeasureName, BFactor, max_split, counter, max_depth, DONE, flg)
                    sub_tree[question].append(answer)

                    sub_tree2[node].append(sub_tree)
                    pre_value = value
                else:
                    data_equal = split_data2(data, split_column, value)
                    if len(data_equal) == 0:
                        classification = classify_data(data, ClassLabel)
                        return classification

                    question = "{} = {}".format(feature_name, value)
                    sub_tree = {question: []}

                    answer = decision_tree_algorithm(data_equal, ClassLabel, MeasureName, BFactor, max_split, counter, max_depth, DONE)
                    sub_tree[question].append(answer)
                    sub_tree2[node].append(sub_tree)
            if feature_name in DONE:
                DONE.remove(feature_name)
            return sub_tree2

# Classifying single instance of the dataset
# For binary DT
def classify_example(example, tree):
    question = list(tree.keys())[0]
    #print(question)
    feature_name, comparison_operator, value = question.split(" ")

    if comparison_operator == "<=":
        if example[feature_name] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]
    else:
        if str(example[feature_name]) == value:
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    if not isinstance(answer, dict):
        return answer

    else:
        residual_tree = answer
        return classify_example(example, residual_tree)

# For non binary DT
def classify_example2(example, tree):
    check = list(tree.keys())[0]
    if '=' not in check:
        ans = []
        tmp = []
        for i in range(len(tree[check])):
            residual_tree = tree[check][i]
            tmp.append(classify_example2(example, residual_tree))
            if not isinstance(tmp[i], dict) and tmp[i] != 'x':
                return tmp[i]
            if tmp[i] != 'x':
                ans.append(tmp[i])
        return

    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split(" ")
    if comparison_operator == "<=":
        if example[feature_name] <= float(value):
            answer = tree[question][0]
            if not isinstance(answer, dict) and answer != 'x':
                return answer
            else:
                residual_tree = answer
                return classify_example2(example, residual_tree)
        else:
            return 'x'
    else:
        if str(example[feature_name]) == value:
            answer = tree[question][0]
            if not isinstance(answer, dict) and answer != 'x':
                return answer
            else:
                residual_tree = answer
                return classify_example2(example, residual_tree)
        else:
            return 'x'

# Calculate the overall accuracy of the Decision Tree Classifier
def calculate_accuracy(df, tree, ClassLabel, BFactor):
    #ClassLabelx = get_ClassLabel(df, ClassLabel)
    df2 = df.copy()
    classification = []
    for i in range(len(df2)):
        if BFactor == 'b':
            classification.append(classify_example(df2.iloc[i], tree))
        else:
            classification.append(classify_example2(df2.iloc[i], tree))

    r = len(df2.columns)
    df2.insert(loc=r, column='classification', value=classification)
    comparison_column = list(np.where(df2['classification'] == df2[ClassLabel], True, False))
    r2 = len(df2.columns)
    df2.insert(loc=r2, column='comparison_column', value=comparison_column)
    '''
    cnt = 0
    for i in range(len(df2)):
        if df2.iloc[i, ClassLabelx] == df2.iloc[i, r]:
            cnt += 1
    accuracy1 = cnt / len(df2)
    '''
    accuracy = df2["comparison_column"].mean()
    return accuracy

# Replace the blank space with '-' from column values
def replace_space(value):
    for i in range(len(value)):
        if value[i] == ' ':
            value = value[:i]+'_'+value[i+1:]
    return value

# :p
def halka_preprocessing(df):
    df.columns = df.columns.str.replace(' ', '_')

    for column in df.columns:
        unique_values = df[column].unique()
        example_value = unique_values[0]

        if isinstance(example_value, str):
            df[column] = df[column].apply(replace_space)
    return df

def get_df_ClassLabel(df):
    list1 = np.array(df.columns)
    ClassLabel = str(list1[-1:])
    ClassLabel = ClassLabel[2:-2]
    return ClassLabel

def ConstructDT(MeasureName, FileName, BFactor, TestFileName, Ratio, ClassLabel, NodeOrder):
    #print("Here goes the Decision Tree construction algorithm with", FileName, " and Measure ", MeasureName, " and the branching Factor ", BFactor)
    # call the functions as given in the command
    '''print(MeasureName, FileName, BFactor, TestFileName, Ratio, ClassLabel, NodeOrder)
    print()'''

    df = pd.read_csv(FileName)
    df = halka_preprocessing(df)
    print(df.columns)

    '''print(df.info())
    print()
    print("The shape of the dataset: ", df.shape)
    print()'''
    # Encode_Label(df)
    # x = df.drop(columns=[ClassLabel])
    # y = df[ClassLabel]
    # print(df.columns)
    # print(ClassLabel)

    if FileName == "credit_card.csv":
        df = df.drop(columns=['Tid'])
    elif FileName == "Iris.csv":
        df = df.drop(columns=['Id'])
    elif FileName == "strange.csv":
        df = df.drop(columns=['gender', 'race/ethnicity', 'lunch'])

    if ClassLabel == None:
        ClassLabel = get_df_ClassLabel(df)
    else:
        if ClassLabel not in df.columns:
            print("Invalid ClassLabel or FileName!")
            return

    if TestFileName == None:
        train_df, test_df = train_test_split(df, test_size=Ratio, random_state=3)
    else:
        test_df = pd.read_csv(TestFileName)
        train_df = df

    max_depth = len(df.columns) - 1
    unique_values = df[ClassLabel].unique()
    #print(len(unique_values))
    if len(unique_values) > 3:
        max_split = len(unique_values)
    else:
        max_split = 3

    if BFactor != 'b':
        print("Maximum branch split for this DT is: ", max_split)
        print()

    if BFactor == 'b':
        accuracy1 = []
        accuracy2 = []
        mx = -1
        for i in range(3):
            tree = decision_tree_algorithm(train_df, ClassLabel, MeasureName=MeasureName, BFactor=BFactor, flg=i)
            if isinstance(tree, dict):
                tmp1 = calculate_accuracy(train_df, tree, ClassLabel, BFactor)
                accuracy1.append(tmp1)
                tmp2 = calculate_accuracy(test_df, tree, ClassLabel, BFactor)
                accuracy2.append(tmp2)
                if tmp2>mx:
                    final_tree = tree
                    mx = tmp2

        print("The DT in dictionary format: ")
        print(final_tree)
        print()
        print("Training Accuracy score: ", max(accuracy1))
        print()
        print("Testing Accuracy score: ", max(accuracy2))
        print()
        if NodeOrder != None:
            print("The DT in Pre Order format: ")
            node = BinaryTree.Node('Null')
            node = make_binary_tree(tree, node)
            node.preorder(node)
            print('End')
            print()

        '''if NodeOrder == 'pre':
            node.preorder(node)
            print('End')
        elif NodeOrder == 'post':
            node.postorder(node)
            print('End')
        else:
            node.inorder(node)
            print('End')'''
    else:
        accuracy1 = []
        accuracy2 = []
        mx = -1
        for i in range(3):
            tree = decision_tree_algorithm(train_df, ClassLabel, max_split=max_split, MeasureName=MeasureName,
                                           BFactor=BFactor, max_depth=max_depth, flg=i)
            if isinstance(tree, dict):
                tmp1 = calculate_accuracy(train_df, tree, ClassLabel, BFactor)
                accuracy1.append(tmp1)
                tmp2 = calculate_accuracy(test_df, tree, ClassLabel, BFactor)
                accuracy2.append(tmp2)
                if tmp2 > mx:
                    final_tree = tree
                    mx = tmp2

        print("The DT in dictionary format: ")
        print(final_tree)
        print()
        print("Training Accuracy score: ", max(accuracy1))
        print()
        print("Testing Accuracy score: ", max(accuracy2))
        print()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # total arguments
    n = len(sys.argv)
    #print(n, "The 1st Param", sys.argv[0])
    MeasureName = "G"
    FileName = "Iris.csv"
    #FileName = "credit_card.csv"
    BFactor = "b"
    TestFileName = None
    Ratio = 0.1
    df = pd.read_csv(FileName)
    ClassLabel = None
    CategoricalColumn = df.iloc[:, -1:]  # df[df.columns[-1]]
    DiscreteColumn = df.iloc[:, 0:1]
    ContinuousColumn = df.iloc[:, 1:2]
    NodeOrder = None # pre-post-in
    # name = sys.argv[1]
    # parsing the command line arguments and setting the parameters
    flg2 = 1
    if n > 1:
        for i in range(1, n):
            param = sys.argv[i]
            if param == "-m" and i+1<n:
                MeasureName = sys.argv[i+1]
                if (MeasureName != "G") and (MeasureName != "E"):
                    print("Invalid Measure Name!")
                    print()
                    MeasureName = "G"
            if param == "-f" and i+1<n:
                FileName = sys.argv[i+1]
                path = pathlib.Path(FileName)
                if not path.exists():
                    print("Invalid File Name!")
                    print()
                    FileName = "Iris.csv"
                #filePath = os.path.abspath(path)
                '''df = pd.read_csv(FileName)
                print(df.info())
                print("The shape of the dataset: ", df.shape)'''
            if param == "-br" and i+1<n:
                BFactor = sys.argv[i+1]
                if BFactor != 'b' and BFactor != 'nb':
                    print("Invalid Branching Factor!")
                    print()
                    BFactor = "b"
            if param == "-t" and i+1<n:
                TestFileName = sys.argv[i+1]
                path = pathlib.Path(TestFileName)
                if not path.exists():
                    print("Invalid Test File Name!")
                    print()
                    TestFileName = None
            if param == "-r" and i+1<n:
                Ratio = 1.0-float(sys.argv[i+1])
                Ratio = round(Ratio,1)
                if Ratio > 1 and i+1<n:
                    print("Invalid Split Ratio!")
                    print()
                    Ratio = 0.1
            if param == "-c" and i+1<n:
                ClassLabel = str(sys.argv[i+1])

            if param == "-d":
                NodeOrder = "pre"
                '''NodeOrder = str(sys.argv[i+1])
                if NodeOrder not in ['pre', 'post', 'in']:
                    print("Invalid Value .")
                    NodeOrder = "pre"'''
            if ('-' in param) and (param not in ["-m", "-f", "-br", "-t", "-r", "-c", "-d"]):
                #print(param)
                #flg2=0
                print("Invalid Switch Parameter!")
                print()
                #break
    if(flg2):
        ConstructDT(MeasureName, FileName, BFactor, TestFileName, Ratio, ClassLabel, NodeOrder)


