import pandas as pd
import numpy as np
import math
from bokeh.models import ColumnDataSource, CategoricalColorMapper
from bokeh.plotting import figure, show

def plot(data_source, gender):
    p = figure(plot_width = 300, plot_height = 300)
    p.circle( source=data_source, x='height', y='weight',color={'field': 'gender', 'transform': color_mapper})
    return p

def male_female_data(height, weight, gender):
    male = []
    female = []
    for h, w, g in zip(height, weight, gender):
        if g == '-1':
            male.append((h, w))
        elif g == '1':
            female.append((h, w))
    return male, female    

def centroid(data):
    sum_x = np.sum( data[:, 0])
    sum_y = np.sum( data[:, 1])
    return (sum_x/len(data), sum_y/len(data))

def radius(centroid, data):
    radius = 0.0
    for point in data:
        distance = math.sqrt( ((centroid[0]-point[0])**2)+((centroid[1]-point[1])**2) )
        if distance > radius:
            radius = distance
    return radius

def generate_hyperplane(b, w):
    point1 = (0, -b/w[1])
    point2 = (-b/w[0], 0)
    return [point1, point2]

# sign = lambda number: (number>0) - (number<0)

def train(height, weight, w, b):
    classifications = []
    for h_value, w_value in zip(height, weight):
        x = np.array([h_value, w_value])
        distance = (np.matmul(np.transpose(w), x) + b) / np.linalg.norm(w)
        if distance < 0:
            classifications.append('-1')
        else:
            classifications.append('1')
    return classifications

def test(height, weight, w, b):
    classifications = []
    for h_value, w_value in zip(height, weight):
        x = np.array([h_value, w_value])
        distance = (np.matmul(np.transpose(w), x) + b) / np.linalg.norm(w)
        if distance < 0:
            classifications.append('-1')
        else:
            classifications.append('1')
    return classifications

def accuracy(classified, expected):
    correct = 0
    for c, e in zip(classified, expected):
        if c != e:
            correct += 1
    return (correct*100.0)/len(expected)

if __name__ == '__main__':
    
    dataset = pd.read_csv('DWH_Training.csv',header=None)
    height = dataset[1]
    weight = dataset[2]
    gender_strings = pd.Series(map(lambda x : str(x), dataset[3].tolist()))

    test_dataset = pd.read_csv('DWH_test.csv',header=None)
    test_height = test_dataset[1]
    test_weight = test_dataset[2]
    test_gender_strings = pd.Series(map(lambda x : str(x), test_dataset[3].tolist()))

    color_mapper = CategoricalColorMapper(factors=gender_strings.unique(), palette=['red', 'blue'])
    data_source = ColumnDataSource(
        data=dict(
            height = dataset[1],
            weight = dataset[2],
            gender = gender_strings
        )
    )

    p = plot(data_source, color_mapper)

    male, female = male_female_data(height, weight, gender_strings)

    c_plus = centroid(np.array(male))
    c_minus = centroid(np.array(female))

    c_plus_radius = radius(c_plus, male)
    c_minus_radius = radius(c_minus, male)

    p.circle(c_plus[0], c_plus[1], color = 'black')
    p.circle(c_minus[0], c_minus[1], color = 'yellow')

    # p.circle(c_plus[0], c_plus[1], radius=c_plus_radius, color='red', fill_alpha=0)
    # p.circle(c_minus[0], c_minus[1], radius=c_minus_radius, color='red', fill_alpha=0)

    b = (np.linalg.norm(np.array(c_minus)))**2 - (np.linalg.norm(np.array(c_plus))**2)

    w = 2 * (np.subtract(c_plus, c_minus))

    hyperplane = generate_hyperplane(b, w)
    p.line(x=[hyperplane[0][0], hyperplane[1][0]], y=[hyperplane[0][1], hyperplane[1][1]],line_color='black', line_width=2)

    train_classification = train(height, weight, w, b)
    train_accuracy = accuracy(train_classification, gender_strings)

    test_classification = test(test_height, test_weight, w, b)
    test_accuracy = accuracy(test_classification, test_gender_strings)

    print train_accuracy
    print test_accuracy

    show(p)

