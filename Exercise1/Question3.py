import numpy as np
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show, output_file
from bokeh.models import Circle, ColumnDataSource, Line, LinearAxis, Range1d, CategoricalColorMapper
import pandas as pd 

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
p.ray(x=[171, 162, 175], y=[40, 65, 40], length=100, angle=[90, 0, 109],
        angle_units="deg", color=["#ff8d00","#226f16","#000000"], line_width=2)
show(p)