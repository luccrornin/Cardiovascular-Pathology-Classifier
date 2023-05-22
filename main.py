import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# df structure: #sample, Signal reading 1, Signal reading 2
# there are 650 thousand samples.

df = pd.read_csv('100.csv')
print(df.head())

# annotations structure: Time   Sample #  Type  Sub Chan  Num
# there are 2275 annotations.
annotation_cols = ['Time', 'Sample', 'Type', 'Sub', 'Chan', 'Num']
annotations = pd.read_csv('100annotations.csv', usecols=annotation_cols)
# the last 3 columns are not needed, so we need a new df with only the first 3 columns
annotations = annotations.iloc[:, :3]
# print(annotations.head())

sample_index = 0

train_data = []
upper_bound = 0
lower_bound = 0

# Future Notes:
# 1. The training input does not include the channel values for the sample that is annotated.
# 2. The channel readings are not float point numbers, this should be changed to ensure no loss of info.
# 3. All of these are lists and will first need to be converted to numpy arrays before appedning to the train_data list.
# 4. The training data will also need to be converted into a numpy array before being fed into the model.
for i in range(len(annotations)):
    sample_row_x = []
    sample_row_y = []
    upper_bound = annotations.iloc[i]['Sample']
    for j in range(lower_bound, upper_bound):
        sample_row_x.append((df.iloc[j][1], df.iloc[j][2]))
        if j == upper_bound - 1:
            sample_row_y.append(annotations.iloc[i]['Type'])

    train_data.append([sample_row_x, sample_row_y])
    lower_bound = upper_bound + 1

print(f'the training data is currently {train_data[1]}')
# print(f'the training samples is {len(train_data[1][0])}')


def count_classe_instances(train_data):
    classes_counts = [0, 0]
    for i in range(len(train_data)):
        match train_data[i][1][0]:
            case 'N':
                classes_counts[0] += 1
            case 'A':
                classes_counts[1] += 1

    return classes_counts


def plot_class_distribution(plot_data):
    x_axis = ['N', 'A']
    plt.bar(x_axis, plot_data)
    plt.xlabel('Class')
    plt.ylabel('Number of Instances')
    plt.title('Class Distribution')
    plt.show()


plot_data = count_classe_instances(train_data)
plot_class_distribution(plot_data)
