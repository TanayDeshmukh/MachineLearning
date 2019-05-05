
# coding: utf-8

# In[94]:



import numpy as np
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show, output_file
from bokeh.models import Circle, ColumnDataSource, Line, Span, LinearAxis, Range1d, CategoricalColorMapper
import pandas as pd 
import math

Dataset = pd.read_csv('DWH_Training.csv',header=None)

gender = pd.Series(map(lambda x : str(x), Dataset[3].tolist()))

data_source = ColumnDataSource(
     data=dict(
         x0=Dataset[1],
         x1=Dataset[2],
         x2=gender
     )
)
color_mapper = CategoricalColorMapper(factors=gender.unique(), palette=['red', 'blue'])
p = figure( plot_width=300, plot_height=300 )
p.circle( source=data_source, x='x0', y='x1',color={'field': 'x2', 'transform': color_mapper})
w = [0.576, 0.047]
b = [-103, -102, -101, -100, -99]
best_accuracy = (0,0)
for i in b:
    p.line([0, -i/w[0]], [-i/w[1], 0], line_width = 2)
    corr = 0
    for index, row in Dataset.iterrows():
        
        dist = ((w[0]*row[1] + w[1]*row[2])+i)/math.sqrt(w[0]*w[0]+w[1]*w[1])
        if((dist<0 and row[3]>0) or (dist>0 and row[3]<0)):
            corr = (corr + 1)
    if((corr*100.0/len(Dataset))>best_accuracy[0]):
        best_accuracy = ((corr*100.0/len(Dataset)), i)
        
print("Accuracy Train Data Set : ",best_accuracy)
show(p)

Dataset_train = pd.read_csv('DWH_test.csv',header=None)
corr = 0
for index, row in Dataset_train.iterrows():
    dist = ((w[0]*row[1] + w[1]*row[2])+best_accuracy[1])/math.sqrt(w[0]*w[0]+w[1]*w[1])
    if((dist<0 and row[3]>0) or (dist>0 and row[3]<0)):
        corr = (corr + 1)
print("Accuracy Test Data Set : ",corr*100.0/len(Dataset_train))



