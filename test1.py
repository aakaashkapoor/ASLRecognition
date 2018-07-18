from os import listdir
import numpy as np
from collections import defaultdict
from sklearn import svm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from sklearn import preprocessing
# #
# from random import random
# from numpy import array
# from numpy import cumsum
# from keras.models import Sequential
# from keras.layers import LSTM
# from keras.layers import Dense
# from keras.layers import TimeDistributed

classList = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r' , 's', 't', 'u', 'v', 'w', 'x', 'y','z',  '1', '2', '3', '4', '5', '6' , '7', '8', '9']


DIR_NAME = "0508-2018"
DIR_SEPARATOR = "/"

rawData = defaultdict(lambda : defaultdict(list)) # a dictionary by gesture name, then by file number
directoryName = defaultdict(lambda : str)


# gets the raw data from the directory
def getRawData(directoryTitle):
    # going through the directory
    directoryCounter = 0  # keep track of the directories
    for directory in listdir(directoryTitle):

        fileCounter = 0  # keep track of the files
        directoryName[directoryCounter] = directory # keeping an index to name tabbin

        # only checking for normal directories
        if("." not in directory):

            # going through all the files in the particular directory
            for filename in listdir(directoryTitle + DIR_SEPARATOR + directory):

                # opens the file in the required format
                # print DIR_NAME + DIR_SEPARATOR + directory + DIR_SEPARATOR + filename
                file = open(directoryTitle + DIR_SEPARATOR + directory + DIR_SEPARATOR + filename, "read")
                temp = file.readlines()

                # Sensor files
                if("SENSOR" in  filename and "_" not in filename):
                    temp = [float(x) for x in temp] # simple parsing
                    # adding to the 3d array
                    rawData[directory][fileCounter] = temp
                    # print "Name : " , directory[0], ", Number : ", directoryCounter, " - File Name : ", filename, " - Length : ", len(temp)
                    fileCounter += 1
                elif("_" not in filename):
                    for temp2 in temp:
                        temp2 = temp2.split("\t")

                        temp2 = [float(x) for x in temp2]  # simple parsing

                        temp2 = temp2[2:]

                        # print "Name : ", directory[0], ", Number : ", directoryCounter, " - File Name : ", filename, " - Length : ", len(temp2)

                        rawData[directory][fileCounter] = temp2

                        fileCounter += 1
            directoryCounter += 1
    # print directoryCounter


#
# get the data of a workbook into
#
def getData2(upper):

    # storing the train and test data
    xTrainData = [[]]
    yTrainData = []
    xTestData = [[]]
    yTestData = []

    # going through all the sheets
    for directory in listdir(DIR_NAME): # ignoring 'R' for now since insignificant data

        if ("." not in directory):
            trainDataList = rawData[directory]

            for data in range(0, upper):
                toAdd = []
                for file in range(0, 11):
                    toAdd += [trainDataList[file][data]]
                xTrainData += [toAdd]
                yTrainData += [classList.index(directory[0])]
                # print toAdd
                # print classList.index(directory[0])

            for data in range(75, 85):
                toAdd = []
                for file in range(0, 11):
                    toAdd += [trainDataList[file][data]]
                xTestData += [toAdd]
                yTestData += [classList.index(directory[0])]
                # print toAdd
                # print classList.index(directory[0])

    return xTrainData[1:], xTestData[1:], yTrainData, yTestData


# model that trains whether the data is [j or z] or it is not
def trainMovementModel(xTrainData, yTrainData2):

    # go through all elements in yTrain and convert it to relevent data for this model
    for index, element in enumerate(yTrainData):

        # if element is either j or z
        if(element == classList.index('j') or element == classList.index('z')):
            yTrainData2[index] = 1
            print'j-z- ', xTrainData[index]

        # if element is not j or z
        else:
            yTrainData2[index] = 0
            print'not j-z- ', xTrainData[index]

    print "STARTED TRAINING MOVEMENT BINARY MODEL"
    # create the model
    # clfMovement = svm.SVC(kernel = 'linear')
    clfMovement = svm.SVC(kernel='poly', degree = 4)
    clfMovement.fit(xTrainData, yTrainData2)

    print "FINISHED TRAINING MOVEMENT BINARY MODEL"

    # save and return the trained model
    joblib.dump(clfMovement, 'clfMovement.pkl')
    return clfMovement


# model that trains whether the data is j or z
def trainJZModel(xTempTrainData, yTempTrainData):

    xTrainData = [[]]
    yTrainData = []


    # go through all elements and get the ones with j and z
    for index, element in enumerate(yTempTrainData):

        # if element is either j or z
        if(element == classList.index('j') or element == classList.index('z')):
            xTrainData += [xTempTrainData[index]]
            yTrainData += [element]

    # ignoring the first element
    xTrainData = xTrainData[1:]


    print "STARTED TRAINING JZ BINARY MODEL"
    clfJZ = svm.SVC(kernel='poly', degree = 1)
    clfJZ.fit(xTrainData, yTrainData)

    print "FINISHED TRAINING JZ BINARY MODEL"

    # save and return the JZ Model
    joblib.dump(clfJZ, 'clfJZ.pkl')
    return clfJZ


# model that trains for the rest of letters
def trainModel(xTempTrainData, yTempTrainData):

    xTrainData = [[]]
    yTrainData = []

    # go through all elements and get the ones with j and z
    for index, element in enumerate(yTempTrainData):

        # if element is either j or z
        if(element != classList.index('j') and element != classList.index('z')):
            xTrainData += [xTempTrainData[index]]
            yTrainData += [element]

    # ignoring the first element
    xTrainData = np.array(xTrainData[1:])

    print "STARTED TRAINING MODEL"
    clf = svm.SVC(kernel = 'linear')
    # clf = svm.SVC(kernel='poly', degree=3)
    clf.fit(xTrainData, yTrainData)

    print "FINISHED TRAINING MODEL"

    # save and return the full model
    joblib.dump(clf, 'clf.pkl')
    return clf

# test the [j or z] or the rest model
def testMovementModel(clfMovement, xTestData, yTrainData):


    # go through all elements in yTrain and convert it to relevent data for this model
    for index, element in enumerate(yTrainData):

        # if element is either j or z
        if(element == classList.index('j') or element == classList.index('z')):
            yTrainData[index] = 1

        # if element is not j or z
        else:
            yTrainData[index] = 0

    print clfMovement.score(xTestData, yTrainData)
    return clfMovement.predict(xTestData)

# test the model if its j or z
def testJZModel(clfJZ, xTempTestData, yTempTestData):

    xTestData = [[]]
    yTestData = []

    # go through all elements and get the ones with j and z
    for index, element in enumerate(yTempTestData):

        # if element is either j or z
        if(element == classList.index('j') or element == classList.index('z')):
            xTestData += [xTempTestData[index]]
            yTestData += [element]

    # ignoring the first element
    xTestData = xTestData[1:]

    print clfJZ.score(xTestData, yTestData)


# test the complete model
def testModel(clf, xTempTestData, yTempTestData):

    xTestData = [[]]
    yTestData = []

    # go through all elements and get the ones with j and z
    for index, element in enumerate(yTempTestData):

        # if element is either j or z
        if(element != classList.index('j') and element != classList.index('z')):
            xTestData += [xTempTestData[index]]
            yTestData += [element]

    # ignoring the first element
    xTestData = np.array(xTestData[1:])

    print clf.score(xTestData, yTestData)

    return clf.predict(xTestData)



# test the complete model
def testCompleteModel(clf, clfMovement, clfJZ, xTestData, yTestData):

    # first predict if they are movement gesture or not
    out = clfMovement.predict(xTestData)

    jzInp = [[]]
    jzInd = []
    Inp = [[]]
    Ind = []

    # separate out the predicted [j or z] from the rest of the letters
    for index, element in enumerate(out):

        if element == 1:
            jzInp += [xTestData[index]]
            jzInd += [index]
        else:
            Inp += [xTestData[index]]
            Ind += [index]

    # if no jz predicted
    if(jzInp == [[]]):
        jzOut = []


    # predict whether j or z
    else:
        jzInp = np.array(jzInp[1:])
        jzOut = clfJZ.predict(jzInp)

    # Then predict with the normal model
    Inp = Inp[1:]
    Inp = np.array(Inp)
    Out = clf.predict(Inp)

    predictedY = yTestData[:]

    # Then put them in the right place
    for index in range(len(jzOut)):
        predictedY[jzInd[index]] = jzOut[index]

    for index in range(len(Out)):
        predictedY[Ind[index]] = Out[index]

    print "ACCURACY =  ",accuracy_score(yTestData, predictedY)

    return predictedY


getRawData("0508-2018")
xTrainData, xTestData, yTrainData, yTestData = getData2(70) # start with 5 or 10

clfJZ = trainJZModel(xTrainData, yTrainData)
# clf = trainModel(xTrainData, yTrainData)
# clf = joblib.load("clf.pkl")
clfMovement = trainMovementModel(xTrainData, yTrainData)
predictedY = testCompleteModel(clf, clfMovement, clfJZ, xTestData, yTestData)



# # clf = trainModel(xTrainData, yTrainData)
# a = testModel(clf, xTestData, yTestData)
# for i in a:
#     print i


# clfJZ = joblib.load("clfJZ.pkl")
# # clfJZ = trainJZModel(xTrainData,yTrainData)
# a = testJZModel(clfJZ, xTestData, yTestData)


# # clfMovement = trainMovementModel(xTrainData,yTrainData)
# clfMovement = joblib.load("clfMovement.pkl")
# a = testMovementModel(clfMovement, xTestData, yTestData)
# print a

# for i in range(len(a)):
#     if a[i] == 1:
#         print i


# # # gets the data
# # # xTrainData, xTestData, yTrainData, yTestData = getData()
# # # # # using polynomial degree 4 because it gives best results
# # clf = svm.SVC(kernel='poly', degree = 1)
# # # #
# # # # print "ready to do this"
# # # # # gets the accuracy
# # clf.fit(xTrainData, yTrainData)
# # print "ACCURACY SCORE FOR ", 70, " items: ", clf.score(xTestData, yTestData)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    # plt.savefig('ConfusionMatrix.png')
#
# upperTraining = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60]
# cnf_matrix = confusion_matrix(yTestData, predictedY)
# for j in range(0, len(cnf_matrix[9])):
#     cnf_matrix[9][j] = 0
# for j in range(0, len(cnf_matrix[25])):
#     cnf_matrix[25][j] = 0
# cnf_matrix[9][9] = 100
# cnf_matrix[25][25] = 100
#
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=classList,
#                       title='Confusion matrix of mixed user training and testing')
# def getAccuracies():
#
#     accuracies = []
#     for i in upperTraining:
#         xTrainData, xTestData, yTrainData, yTestData = getData2(i)
#         # gets the data
#         # xTrainData, xTestData, yTrainData, yTestData = getData()
#         # # # using polynomial degree 4 because it gives best results
#         clf = svm.SVC(kernel='poly', degree = 1)
#         # #
#         # # print "ready to do this"
#         # # # gets the accuracy
#         clf.fit(xTrainData, yTrainData)
#         print "ACCURACY SCORE FOR ", i, " items: ", clf.score(xTestData, yTestData)
#         accuracies += [clf.score(xTestData, yTestData)]
#     return accuracies
# # accuracies += [clf.score(xTestData, yTestData)]
# # print accuracies
# # # cnf_matrix = confusion_matrix(yTestData, clf.predict(xTestData))
# # #


# accuracies = [x*100 for x in accuracies]
# upperTraining = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
# plt.plot(upperTraining, accuracies, '-bo', color = 'r')
# plt.ylabel('Accuracy of the classification in percentage')
# plt.xlabel('Amount of training data for each gesture')
# plt.xlim([0,80])
# plt.ylim([20,100])
# plt.grid()
# plt.title('Accuracies of Different Amount of Data using SVM')
# plt.show()
# plt.savefig('graph.png')


###### EXTRA CODE #########
# # for mixed training and testing
# getRawData("0502-2018")
# xTrainData2, xTestData2, yTrainData2, yTestData2 = getData2(70)
#
# xTrainData = xTrainData + xTrainData2
# yTrainData = yTrainData + yTrainData2
# xTestData = xTestData + xTestData2
# yTestData = yTestData + yTestData2

# X = xTestData + xTrainData
# X = np.array(X)
# X = normalize(X, norm = 'max', axis = 0)

# meanList =[]
# stdList = []
#
# for feature in range(0,11):
#     meanList += [np.mean(X[:,feature])]
#     stdList += [np.std(X[:,feature])]
#
# for feature in range(0,11):
#     for row in range(len(X)):
#         X[row][feature] = (X[row][feature] - meanList[feature])/ stdList[feature]
#
# print(meanList)
# print(stdList)
# xTestData = X[:len(xTestData)]
# xTrainData = X[len(xTestData):]
# xTrainData = np.array(xTrainData)
# xTestData = np.array(xTestData)

# clfMovement = joblib.load('clfMovement.pkl')
# clfJZ = joblib.load('clfJZ.pkl')
# clf = joblib.load('clf.pkl')