import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sample_index = 0


# Future Notes:
# 1. The training input does not include the channel values for the sample that is annotated.
# 2. The channel readings are not float point numbers, this should be changed to ensure no loss of info.
# 3. All of these are lists and will first need to be converted to numpy arrays before appedning to the train_data list.
# 4. The training data will also need to be converted into a numpy array before being fed into the model.
def read_in_csv(mv_file, annotations_file):
    """
    This function reads in 2 csv files and returns 1 dataframe.
    :param mv_file: The csv file containing the mv readings for the whole ambulatory ECG scan.
    :param annotations_file: The csv file containing the annotations for the ambulatory ECG scan.
    :return: The returned data frame consists of the
    timeseries of a heartbeat, followed by the annotations of the heartbeat. The time series consists of a list of
    tuples with two values, one for each channel.
    """
    # mv_readings structure: #sample, Signal reading 1, Signal reading 2
    # there are 650k samples.
    mv_readings = pd.read_csv(mv_file)

    # annotations structure: Time   Sample #  Type  Sub Chan  Num
    # there are 2275 annotations.
    annotations_cols = ['Time', 'Sample', 'Type', 'Sub', 'Chan', 'Num']
    annotations = pd.read_csv(annotations_file, usecols=annotations_cols)
    # the last 3 columns are not needed, so we need a new df with only the first 3 columns
    annotations = annotations.iloc[:, :3]

    train_data = []
    upper_bound = 0
    lower_bound = 0
    for i in range(len(annotations)):
        sample_row_x = []
        sample_row_y = []
        upper_bound = annotations.iloc[i]['Sample']
        for j in range(lower_bound, upper_bound):
            sample_row_x.append((mv_readings.iloc[j][1], mv_readings.iloc[j][2]))
            if j == upper_bound - 1:
                sample_row_y.append(annotations.iloc[i]['Type'])

        train_data.append([sample_row_x, sample_row_y])
        lower_bound = upper_bound + 1

    print(f'the training data is currently {train_data[1]}')
    return train_data


def count_classe_instances(data):
    """
    This function counts the number of instances of each type of heartbeat in the data.
    :param data: This is the collection of data after preprocessing.
    :return: returns a list as long as there are types of heartbeats. Each index corresponding to the instance of a type
    """
    classes_counts = [0, 0]
    for i in range(len(data)):
        if data[i][1][0] == 'N':
            classes_counts[0] += 1
        elif data[i][1][0] == 'A':
            classes_counts[1] += 1

    return classes_counts


def plot_class_distribution(plot_data):
    """
    This function plots the heartbeat type distribution of the data.
    :param plot_data: the instances of each type of heartbeat.
    :return: None
    """
    x_axis = ['N', 'A']
    plt.bar(x_axis, plot_data)
    plt.xlabel('Class')
    plt.ylabel('Number of Instances')
    plt.title('Class Distribution')
    plt.show()


def main():

    mv_file = '100.csv'
    annotation_file = '100annotations.csv'

    train_data = read_in_csv(mv_file, annotation_file)

    plot_data = count_classe_instances(train_data)
    plot_class_distribution(plot_data)


if __name__ == '__main__':
    main()
