from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, CategoricalColorMapper
import pandas as pd
import math

def hyperPlanes(w, b):
    hyperplanes=[]
    for b_value in b:
        x = (-b_value)/w[0]
        y = (-b_value)/w[1]
        hyperplanes.append([(x, 0),(0, y)])

    return hyperplanes

def plotGraph(data_source, color_mapper, hyperPlanes):
    p = figure( plot_width=300, plot_height=300 )
    p.circle( source=data_source, x='x0', y='x1',color={'field': 'x2', 'transform': color_mapper})
    for plane in hyperPlanes:
        p.line(x = plane[0], y = plane[1], line_width=1)
    show(p)

def computeSignedDistance(w, b, x):
    distance =  ((((w[0] * x[0]) + (w[1] * x[1])) + b )) / math.sqrt(math.pow(w[0], 2) + math.pow(w[1], 2))
    return distance

sign = lambda number: (number>0) - (number<0)

def trainData(height, weight, gender, w, b):
    accuracyValues = []
    for b_value in b:
        correctClassificationCount = 0
        for (x, y, g) in zip(height, weight, gender):
            signedDistance = computeSignedDistance(w, b_value, [x, y])
            if(sign(signedDistance) != g):
                correctClassificationCount += 1 
        accuracy = (correctClassificationCount * 100.0)/len(gender)
        accuracyValues.append((accuracy, b_value))
    return accuracyValues

def testData(height, weight, gender, w, b):
    return trainData(height, weight, gender, w, b)
    # Because it is basically the same algorithm running just using different data
    # Should be done with a different approach. Train and Test should return the output, and 
    # this output should be compared to get the accuracy.

def bestAccuracy(accuracyValues):
    bestAccuracy = 0
    b_value = 0
    for accuracy in accuracyValues:
        if accuracy[0] > bestAccuracy:
            bestAccuracy = accuracy[0]
            b_value = accuracy[1]
    return b_value

if __name__ == '__main__':

    Dataset = pd.read_csv('DWH_Training.csv',header=None)
    testDataset = pd.read_csv('DWH_test.csv',header=None)
    height = Dataset[1]
    weight = Dataset[2]
    gender = pd.Series(map(lambda x : str(x), Dataset[3].tolist()))

    w = [0.576, 0.047]
    b = [-103, -102, -101, -100, -99]

    data_source = ColumnDataSource(
                    data=dict(
                        x0=Dataset[1],
                        x1=Dataset[2],
                        x2=gender
                    )
                )
    
    hyperPlanes = hyperPlanes(w, b)
    
    color_mapper = CategoricalColorMapper(factors=gender.unique(), palette=['red', 'blue'])
    plotGraph(data_source, color_mapper, hyperPlanes)
    
    # accuracyValues = trainData(height, weight, Dataset[3].tolist(), w, b)

    # hyperPlaneClassifyingTheData = bestAccuracy(accuracyValues)

    # print hyperPlaneClassifyingTheData

    # testheight = testDataset[1]
    # testweight = testDataset[2]
    # testgender = pd.Series(map(lambda x : str(x), testDataset[3].tolist()))

    # accuracyValues = testData(testheight, testweight, testDataset[3].tolist(), w, [hyperPlaneClassifyingTheData])

    # print accuracyValues
