import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
import test


def buter(mv_file):
    """
    This function implements a butterworth filter with
    an order of 3,
    a low cut-off of 0.4Hz,
    a high cut-off of 45Hz,
    a sampling rate of 360Hz.
    :return: The filtered data in a numpy array.
    """
    order = 3
    low_cut = 0.4
    high_cut = 45
    sampling_rate = 360
    nyq = 0.5 * sampling_rate
    low = low_cut / nyq
    high = high_cut / nyq
    coef = butter(order, [low, high], btype='band')
    a = coef[0]
    b = coef[1]

    df = pd.read_csv(mv_file)
    index = np.asarray(df.iloc[:, 0])
    mv_readings = np.asarray(df.iloc[:, 1:])
    filtered_data = filtfilt(a, b, mv_readings, axis=0)
    filtered_df = np.insert(filtered_data, 0, index, axis=1)

    return filtered_df


# Future Notes:
# 1. The training input does not include the channel values for the sample that is annotated.
# 2. The channel readings are not float point numbers, this should be changed to ensure no loss of info.
# 3. All of these are lists and will first need to be converted to numpy arrays before appedning to the train_data list.
# 4. The training data will also need to be converted into a numpy array before being fed into the model.
def old_read_in_csv(mv_file, annotations_file):
    """
    This function reads in 2 csv files and returns 1 dataframe.
    :param mv_file: The csv file containing the mv readings for the whole ambulatory ECG scan.
    :param annotations_file: The csv file containing the annotations for the ambulatory ECG scan.
    :return: The dataframe containing the mv readings and the annotations as a numpy array & a list of the types.
    Data Frame Structure:
    The returned data frame consists of [sample_index][mv_readings = 0|annotation = 1][tuples of channels]
    [Channel 1 or 2|Type] & the types of heartbeats in a list.
    """
    # mv_readings structure: #sample, Signal reading 1, Signal reading 2
    # there are 650k samples.
    buttery_df = buter(mv_file)  # This is the filtered data, using the butterworth filter.
    mv_readings = pd.read_csv(mv_file)

    # annotations structure: Time   Sample #  Type  Sub Chan  Num
    # there are 2275 annotations.
    annotations_cols = ['Time', 'Sample', 'Type', 'Sub', 'Chan', 'Num']
    annotations = pd.read_csv(annotations_file, usecols=annotations_cols)
    # the last 3 columns are not needed, so we need a new df with only the first 3 columns
    annotations = annotations.iloc[:, :3]

    train_data = []
    heartbeat_types = []
    upper_bound = 0
    lower_bound = 0
    for i in range(len(annotations)):
        sample_row_x = []  # This is the time series of a heartbeat
        sample_row_y = []  # This is the annotation of the heartbeat
        upper_bound = annotations.iloc[i]['Sample']
        # if we see a type of heartbeat that we have not seen before, we add it to the list of heartbeat types
        if annotations.iloc[i]['Type'] not in heartbeat_types:
            heartbeat_types.append(annotations.iloc[i]['Type'])

        sample_row_y.append(annotations.iloc[i]['Type'])

        # Run through the mv readings and append the readings to the sample_row_x as tuples of (channel 1, channel 2)
        for j in range(lower_bound, upper_bound):
            sample_row_x.append((buttery_df[j][1], buttery_df[j][2]))

        train_data.append([sample_row_x, sample_row_y])
        lower_bound = upper_bound + 1

    return train_data, heartbeat_types


def read_in_csv(mv_file, annotations_file):
    # So this currently is a numpy matrix with shape (650_000, 3)
    butter_df = buter(mv_file)

    # annotations structure: Time   Sample #  Type  Sub Chan  Num
    # there are 2275 annotations.
    annotations_cols = ['Time', 'Sample', 'Type', 'Sub', 'Chan', 'Num']
    annotations = pd.read_csv(annotations_file, usecols=annotations_cols)
    # the last 3 columns are not needed, so we need a new df with only the first 3 columns
    # The shape of the annotations is (2_039, 3)
    annotations = annotations.iloc[:, :3]

    print(annotations.iloc[0])

    train_data = []
    label_data = []
    heartbeat_types = []

    upper_bound = 0
    lower_bound = 0
    for i in range(annotations.shape[0]):
        if annotations.iloc[i][0] not in heartbeat_types:
            heartbeat_types.append(annotations.iloc[i][2])

        label_data.append(annotations.iloc[i][2])

        heartbeat_sequence = []
        upper_bound = annotations.iloc[i][1]
        for pair in range(lower_bound, upper_bound):  # The value at the upper bound is not included.
            # inspect later if converting the pair into a np array is necessary.
            channel_readings = [butter_df[pair][1], butter_df[pair][2]]
            heartbeat_sequence.append(np.asarray(channel_readings))

        train_data.append(np.asarray(heartbeat_sequence))
        lower_bound = upper_bound + 1

    return train_data, label_data, heartbeat_types


def compare(new_seq, label, old):
    """
    This is just a quick helper function, wanted to see if the way I structured the data was correct and that all values
    matched up as well as labels
    :param new_seq: The new sequence of heartbeats, this is now only the training data input.
    :param label: This is target labels for the input data.
    :param old: This is the old data structure that was used where the input and label were both present.
    :return:
    """
    if len(new_seq) != len(label):
        raise ValueError("The two lists are not the same length.")

    print(new_seq[0][0][0])
    print('------')
    print(old[0][0][0][0])

    # This layer will get the heartbeat segments
    for i in range(len(new_seq)):
        # this layer will iterate though the values in the segment
        for j in range(len(old[i])):
            if new_seq[i][j][0] != old[i][0][j][0] and new_seq[i][j][1] != old[i][0][j][1]:
                print("The values do not match.")
                return False
            # check if the label matches the label in the old list
            if label[i] != old[i][1][0]:
                print("The labels do not match.")
                return False

    print("The values match and labels match.")


def split_into_types(data, heartbeat_types):
    """
    This function splits the data into the different types of heartbeats.
    :param data: The data to be split.
    :param heartbeat_types: The types of heartbeats in the data.
    :return: A list of lists, where each list is a type of heartbeat.
    """
    # the classes are N, A, a, J, V, F, j, p
    classes = []
    for i in range(len(heartbeat_types)):
        classes.append([])

    for i in range(len(data)):
        beat = data[i][1][0]
        if beat in heartbeat_types:
            classes[heartbeat_types.index(beat)].append(data[i])

    return classes


def count_classe_instances(data, heartbeat_types):
    """
    This function counts the number of instances of each type of heartbeat in the data.
    :param heartbeat_types: The types of heartbeats in the data.
    :param data: This is the collection of data after preprocessing.
    :return: returns a list as long as there are types of heartbeats. Each index corresponding to the instance of a type
    """
    # the classes are N, A, a, J, V, F, j, p
    classes_counts = np.zeros(len(heartbeat_types))

    for i in range(len(data)):
        beat = data[i][1][0]
        if beat in heartbeat_types:
            classes_counts[heartbeat_types.index(beat)] += 1

    return classes_counts


def plot_class_distribution(plot_data, heartbeat_types):
    """
    This function plots the heartbeat type distribution of the data.
    :param heartbeat_types: The types of heartbeats in the data.
    :param plot_data: the instances of each type of heartbeat.
    :return: None
    """
    # the classes which represent our x-axis are N, A, a, J, V, F, j, p
    # x_axis = ['N', 'A', 'a', 'J', 'V', 'F', 'j', 'p']
    plt.bar(heartbeat_types, plot_data)
    plt.xlabel('Heartbeat Types')
    plt.ylabel('Number of Instances')
    plt.title('Heartbeat type Distribution')
    plt.show()


def get_avg_seq_length(train_data):
    """
    This function calculates the average length of the sequences in the training data.
    :param train_data: The training data.
    :return: The average length of the sequences in the training data.
    """
    # Calculate the number of sequences in the training data.
    num_sequences = len(train_data)
    # Calculate the total length of the sequences in the training data.
    total_length = 0
    for sequence in train_data:
        total_length += len(sequence[0])
    # Calculate the average length of the sequences in the training data.
    avg_seq_length = total_length / num_sequences
    # Return the average length of the sequences in the training data.
    return avg_seq_length


def main():
    mv_file = '201.csv'
    annotation_file = '201annotations.csv'
    # the old read in csv function has the target labels in it as well.
    train_data, heartbeat_types = old_read_in_csv(mv_file, annotation_file)
    # The new one does not have the target labels in it and everything is a numpy array.
    train_data, label_data, heartbeat_types = read_in_csv(mv_file, annotation_file)

    # print(len(heartbeat_types))
    # print(train_data[0][0][0])
    # print(f'The average heartbeat segment length is: {get_avg_seq_length(train_data)}')

    # plot_data = count_classe_instances(train_data, heartbeat_types)
    # This matrix contains as many lists as there are types of heartbeats. Each list then has every sample of that type.
    class_split_df = split_into_types(train_data, heartbeat_types)
    # plot_class_distribution(plot_data, heartbeat_types)

    # ------------------------------------------------------------------------------------------------------------------
    # This is the portion of code for building the ESN model.

    Nx = 500
    Nu = 2
    Ny = 10
    alpha = 0.3
    sparseness = 0.1
    little_bound = -1
    big_bound = 1
    # rescale_factor = 0.4

    # create the input weight matrix w_in
    w_in = test.generate_sparse_weights(Nx, Nu, sparseness, little_bound, big_bound)

    # Generate the sparse matrix of weights for the reservoir (W).
    sparse_matrix = test.generate_hidden_matrix(Nx, sparseness, little_bound, big_bound)
    # Calculate the spectral radius of the reservoir weights.
    spectral_radius = test.get_spectral_radius(sparse_matrix)
    # rescale the weights
    w = sparse_matrix / spectral_radius

    # Generate the initial state of the reservoir.
    x = test.generate_neurons(Nx)

    # Generate the weights for the output layer of the ESN model.
    w_out = test.generate_sparse_weights(Nx, Ny, sparseness, little_bound, big_bound)

    # Generate the units for the output layer of the ESN model.
    y = test.generate_neurons(Ny)

    # ------------------------------------------------------------------------------------------------------------------
    # This is the portion of code for training the ESN model.


if __name__ == '__main__':
    main()
