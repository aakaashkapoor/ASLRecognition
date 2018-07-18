# function that takes in trained data of an american

import xlrd
from sklearn import svm


# get the workbook in a variable to get all the sheets one by one
workbook = xlrd.open_workbook('/Users/aakaashkapoor/Downloads/Research/02-26-18 SLR.xlsx')

#
# get the data of a workbook into
#
def getData():

    # storing the incrementing index for train and test
    inputIndex = -1
    inputTestIndex = -1

    # storing the train and test data
    xTrainData = [[]]
    yTrainData = []
    xTestData = [[]]
    yTestData = []

    # going through all the sheets
    for i in range(len(workbook.sheet_names())):
        # getting each worksheet
        worksheet = workbook.sheet_by_index(i)

        # since these have an empty column
        if(i == 18 or i == 31):
            continue

        # to find out the number of rows to train on
        max = ((((worksheet._dimnrows - 3) / 125) - 1)*125) + 2

        # getting all the rows for the training data
        for row in range(2, max):
            # one round of data or 5 seconds
            if ((row - 2) % 125 == 0):
                inputIndex += 1
                yTrainData.append(float(i))
                xTrainData += [[]] # next input layer
            # going through all the columns in the worksheet
            for col in range(1, worksheet.ncols):
                xTrainData[inputIndex] += [float(worksheet.cell_value(row, col))] # adding to the data

        # getting all the rows for the testing data
        for row in range(max, max + 125):
            if ((row - 2) % 125 == 0):  # one round of data or 5 seconds
                inputTestIndex += 1
                yTestData.append(float(i))
                xTestData += [[]]
            # going through all the columns in the worksheet
            for col in range(1, worksheet.ncols):
                xTestData[inputTestIndex].append(float(worksheet.cell_value(row, col))) # adding to the data

    # returning the training and testing arrays
    return normalize(xTrainData[:-1], norm = 'l2', axis = 0), normalize(xTestData[:-1], norm = 'l2', axis = 0), \
           normalize(yTrainData, norm='l2', axis=0), normalize(yTestData, norm = 'l2', axis = 0)


# gets the data
xTrainData, xTestData, yTrainData, yTestData = getData()
# using polynomial degree 4 because it gives best results
clf = svm.SVC(kernel='poly', degree = 4)
# gets the accuracy
print clf.fit(xTrainData, yTrainData).score(xTestData, yTestData)


