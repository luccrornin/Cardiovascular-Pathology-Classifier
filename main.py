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
    buttery_mv_df = buter(mv_file)  # This is the filtered data, using the butterworth filter.

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
            sample_row_x.append((buttery_mv_df[j][1], buttery_mv_df[j][2]))

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

    train_data = []
    label_data = []
    heartbeat_types = []

    upper_bound = 0
    lower_bound = 0
    # annotations.shape[0]
    for i in range(annotations.shape[0]):
        if annotations.iloc[i][2] not in heartbeat_types:
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


def generate_y_target(label_data, heartbeat_types):
    """
    This function will generate the one hot encoding for the labels, which will be the target for the model.
    :param label_data: The list of labels for the training data, each label corresponds to a heartbeat sequence.
    :param heartbeat_types: The list of all the different heartbeat types. The one hot encoding will be based on this.
    :return: A matrix of shape (len(label_data), len(heartbeat_types)) where each row is a
    one hot encoding of the label.
    """
    y_target = np.zeros((len(label_data), len(heartbeat_types)))
    for i in range(len(label_data)):
        y_target[i][heartbeat_types.index(label_data[i])] = 1

    return y_target


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


def split_into_types(data, heartbeat_types, label_data):
    """
    This function splits the data into the different types of heartbeats.
    :param data: The data to be split.
    :param heartbeat_types: The types of heartbeats in the data.
    :param label_data:

    :return: A list of lists, where each list is a type of heartbeat.
    """
    # the classes are N, A, a, J, V, F, j, p
    classes = []
    for i in range(len(heartbeat_types)):
        classes.append([])

    for i in range(len(data)):
        beat = label_data[i]
        if beat in heartbeat_types:
            classes[heartbeat_types.index(beat)].append(data[i])

    return classes


def count_classe_instances(label_data, heartbeat_types):
    """
    This function counts the number of instances of each type of heartbeat in the data.
    :param label_data: The labels for each heartbeat sequence in the training data.
    :param heartbeat_types: The types of heartbeats in the data.
    :return: returns a list as long as there are types of heartbeats. Each index corresponds to the instance of a type.
    """
    # the classes are N, A, a, J, V, F, j, p
    classes_counts = np.zeros(len(heartbeat_types))

    for i in range(len(label_data)):
        beat = label_data[i]
        if beat in heartbeat_types:
            classes_counts[heartbeat_types.index(beat)] += 1

    return classes_counts


def plot_class_distribution(plot_data, heartbeat_types):
    """
    This function plots the heartbeat type distribution of the data.
    :param heartbeat_types: The types of heartbeats in the data.
    :param plot_data: The instances of each type of heartbeat.
    :return: None.
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
    # -------------------------------------------- Data Preprocessing --------------------------------------------------
    mv_file = '201.csv'
    annotation_file = '201annotations.csv'
    # the old read in csv function has the target labels in it as well.
    # train_data, heartbeat_types = old_read_in_csv(mv_file, annotation_file)

    # The new one does not have the target labels in it and everything is a numpy array.
    train_data, label_data, heartbeat_types = read_in_csv(mv_file, annotation_file)

    # class_split_df = split_into_types(train_data, heartbeat_types, label_data)

    # -------------------------------------------- Input Data Visualization --------------------------------------------
    base_input = pd.read_csv(mv_file)
    base_input = np.delete(base_input, 0, axis=1)
    base_input = base_input[:159, :]  # specify the sample to go until, check the annotations file for the sample number

    heartbeat = train_data[0]

    butter = buter(mv_file)
    butter = np.delete(butter, 0, axis=1)
    butter = butter[:159, :]  # specify the sample to go until, check the annotations file for the sample number.

    # test.plot_input_data(base_input)
    #
    # y_target = generate_y_target(label_data, heartbeat_types)

    # print(f'The average heartbeat segment length is: {get_avg_seq_length(train_data)}')

    # plot_data = count_classe_instances(label_data, heartbeat_types)
    # This matrix contains as many lists as there are types of heartbeats. Each list then has every sample of that type.

    # plot_class_distribution(plot_data, heartbeat_types)

    # --------------------------------------------Building the ESN model------------------------------------------------
    # This is the portion of code for .

    Nx = 500
    Nu = 2
    Ny = 10
    alpha = 0.3
    sparseness = 0.1
    little_bound = -1
    big_bound = 1
    # rescale_factor = 0.4

    esn = test.ESN(Nx, Nu, Ny, sparseness, alpha, little_bound, big_bound)
    # esn.train_readout(harvested_states, y_target)
    harvested_states = esn.harvest_state(train_data, len(train_data))

    # --------------------------------------------Operating the ESN model-----------------------------------------------

    # print("The Heartbeat Types are: ", heartbeat_types)
    # esn.set_w_out(np.load('w_out.npy'))
    # test.output_activation_plot(esn, 1, train_data[1])
    num_neurons_to_plot = 4
    num_neurons_per_plot = 2
    num_heartbeats_to_feed = 10
    esn.timeseries_activation_plot(train_data, num_neurons_to_plot, num_neurons_per_plot, num_heartbeats_to_feed, False)



if __name__ == '__main__':
    main()
