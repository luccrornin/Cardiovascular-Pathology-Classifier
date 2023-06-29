import copy
import random

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
import test


def scale_input(input_data, input_min, input_max, output_min, output_max):
    """
    Scale the input data to a specified output range.

    Args:
        input_data (ndarray): Input data to be scaled.
        input_min (float): Minimum value of the input data.
        input_max (float): Maximum value of the input data.
        output_min (float): Desired minimum value of the output range.
        output_max (float): Desired maximum value of the output range.

    Returns:
        ndarray: Scaled input data.
    """
    scaled_data = ((input_data - input_min) / (input_max - input_min)) * (output_max - output_min) + output_min
    return scaled_data


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
    # The columns will be filtered with a butterworth filter.
    butter_df = buter(mv_file)

    # something = np.delete(butter_df, 0, axis=1)
    # something = something[:159 + 1, :]
    # test.plot_input_data(something, "Filtered Data")

    # The MV readings now need to be scaled to be between -1 and 1.
    # We get the overall min & max of the MV readings.
    overall_min = min(np.min(butter_df[:, 1]), np.min(butter_df[:, 2]))
    overall_max = max(np.max(butter_df[:, 1]), np.max(butter_df[:, 2]))

    # Get the columns as numpy arrays.
    mlii = butter_df[:, 1]
    v5 = butter_df[:, 2]

    # Scale the columns.
    mlii = scale_input(mlii, overall_min, overall_max, -1, 1)
    v5 = scale_input(v5, overall_min, overall_max, -1, 1)

    # other = np.column_stack((mlii, v5))
    # other = other[:159 + 1, :]
    # test.plot_input_data(other, "Scaled & Filtered Data")

    # Put the scaled columns back into the butter_df.
    butter_df = np.column_stack((butter_df[:, 0], mlii, v5))

    # annotations structure: Time   Sample #  Type  Sub Chan  Num
    # there are 2,275 annotations.
    # annotations_cols = ['Time', 'Sample #', 'Type', 'Sub', 'Chan', 'Aux']
    annotations = pd.read_csv(annotations_file)

    # the last 3 columns are not needed, so we need a new df with only the first 3 columns
    # The shape of the annotations is (2_039, 3)
    annotations = annotations.iloc[:, :3]

    train_data = []
    label_data = []
    heartbeat_types = []

    upper_bound = 0
    lower_bound = 0

    # This loop iterates loops over the annotation file, giving indications of the labels and the bound of the
    # heartbeats
    for i in range(annotations.shape[0]):
        # If the heartbeat is not in the list of heartbeat types, add it to the list.
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

    print(f'\033[36mThe training data, label data and heartbeat types have been read in.\n')
    return train_data, label_data, heartbeat_types


def balance_classes(train_data, label_data, heartbeat_types):
    """
    This code balances the number of instances of normal beats and abnormal beats.
    :param train_data: The original training data.
    :param label_data:
    :param heartbeat_types:
    :return:
    """
    type_counts = count_classe_instances(label_data, heartbeat_types)
    num_normal_beats_to_remove = int(type_counts[2] - type_counts[3])

    normal_beat_indices = [i for i, val in enumerate(label_data) if val == 'N']
    random_normal_beat_indices = random.sample(normal_beat_indices, num_normal_beats_to_remove)
    for idx in sorted(random_normal_beat_indices, reverse=True):
        del train_data[idx]
        del label_data[idx]

    not_wanted_labels = [i for i, val in enumerate(label_data) if val == '~' or val == '+']
    for idx in sorted(not_wanted_labels, reverse=True):
        del train_data[idx]
        del label_data[idx]

    return train_data, label_data


def re_sample_data(train_data, save=False):
    """
    This function will re-sample the data such that every tenth point will be selected from each heartbeat sequence.
    :param train_data: The collection of all heartbeat segments
    :param save: A boolean indicating whether the re-sampled data should be saved to a file.
    :return: The re-sampled data.
    """
    for i, heartbeat in enumerate(train_data):
        altered_heartbeat = heartbeat[::10]
        train_data[i] = altered_heartbeat

    train_data = np.asarray(train_data)
    if save:
        np.save('re_sampled_data.npy', train_data)
    return train_data


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


def split_into_types(data, heartbeat_types, label_data):
    """
    This function splits the data into the different types of heartbeats.
    :param data: The data to be split.
    :param heartbeat_types: The types of heartbeats in the data.
    :param label_data:

    :return: A list of lists, where each list is a type of heartbeat.
    """
    print(heartbeat_types)
    classes = []
    for i in range(len(heartbeat_types)):
        classes.append([])

    for i in range(len(data)):
        beat = label_data[i]
        if beat in heartbeat_types:
            classes[heartbeat_types.index(beat)].append(data[i])

    reduced_classes = []

    return classes


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
        total_length += len(sequence)
    # Calculate the average length of the sequences in the training data.
    avg_seq_length = total_length / num_sequences
    # Return the average length of the sequences in the training data.
    return avg_seq_length


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


def main():
    # -------------------------------------------- Data Preprocessing --------------------------------------------------
    mv_file = '106.csv'
    annotation_file = '106annotations - 106annotations.csv'
    # the old read in csv function has the target labels in it as well.
    # train_data, heartbeat_types = old_read_in_csv(mv_file, annotation_file)

    # butter_df = buter(mv_file)

    # df = pd.read_csv(annotation_file)
    # subset = df.iloc[::10, :]
    # subset.to_csv('104_subset.csv', encoding='utf-8-sig', index=False)
    #
    # print(subset.head())

    # The new one does not have the target labels in it and everything is a numpy array.
    train_data, label_data, heartbeat_types = read_in_csv(mv_file, annotation_file)
    train_data, label_data = balance_classes(train_data, label_data, heartbeat_types)

    # print(label_data[568])

    # Generate the target one hot encodings the model should output.
    y_target = generate_y_target(label_data, heartbeat_types)

    # -------------------------------------------- Input Data Visualization --------------------------------------------
    # base_input = pd.read_csv(mv_file)
    # base_input = np.delete(base_input, 0, axis=1)
    # heartbeat_sample = 159
    # base_input = base_input[:heartbeat_sample + 1,
    #              :]  # specify the sample to go until, check the annotations file for the sample number
    #
    # heartbeat = train_data[0]
    #
    # butter = buter(mv_file)
    # butter = np.delete(butter, 0, axis=1)
    # butter = butter[:heartbeat_sample + 1,
    #          :]  # specify the sample to go until, check the annotations file for the sample number.
    #
    # # test.plot_input_data(butter, "Butterworth Data, Heartbeat 0 & 1")
    #
    #
    # # print(f'\033[33m The average heartbeat segment length is: {get_avg_seq_length(train_data)}')
    #

    plot_data = count_classe_instances(label_data, heartbeat_types)  # This matrix contains as many lists as there
    # are types of heartbeats. Each list then has every sample of that type.

    plot_class_distribution(plot_data, heartbeat_types)

    # --------------------------------------------Building the ESN model------------------------------------------------
    # This is the portion of code for .

    Nu = 2
    Ny = 7

    Nx = 400
    input_bias = 0.1
    reservoir_bias = 1
    sparseness = 0.1
    little_bound = -0.8
    big_bound = 0.8
    alpha = 0.3
    save = True

    # Create the ESN model.
    # esn = test.ESN(Nx, Nu, Ny, input_bias, reservoir_bias, sparseness, alpha, little_bound, big_bound)
    # esn.train(train_data, y_target)
    # esn.test(train_data, y_target)

    # Here we harvest the activations of the reservoir units at the end of each heartbeat sequence.
    # harvested_states = esn.harvest_state(train_data, len(train_data))

    # Here we train the readout weights of the ESN model using the harvested states and the target labels the model
    # should output.
    # esn.train_readout(harvested_states, y_target, save)

    # --------------------------------------------Operating the ESN model-----------------------------------------------

    # print("The Heartbeat Types are: ", heartbeat_types)
    # esn.set_w_out(np.load('w_out.npy'))
    # debug_output_weights = esn.w_out
    # esn.set_linear_model('linear_model.pkl')

    num_neurons_to_plot = 3
    num_neurons_per_plot = 3
    num_heartbeats_to_feed = 1
    # state_activation_title = f'New Activations spareness = {sparseness}, n_x = {Nx}'
    state_activation_title = f'L-bound = {little_bound}, U-bound = {big_bound}-wee'
    segment_wise = False
    # esn.timeseries_activation_plot([train_data[0]], num_neurons_to_plot, num_neurons_per_plot, num_heartbeats_to_feed,
    #                                state_activation_title, segment_wise)

    # Now that we have output weight matrix we can classify the data.

    # test_beat_1 = train_data[0]
    # test_beat_2 = train_data[1]
    # test_beat_3 = train_data[2]
    # test_beat_4 = train_data[568]
    # esn.classify(test_beat_2)
    # esn.get_y(heartbeat_types)
    # print(f'\033[32m The predicted class is: {the_hotnesss}')


if __name__ == '__main__':
    main()

# selected_neurons = np.random.choice(np.arange(self.n_x), num_neurons, replace=False)
# captured_data = np.empty((num_neurons, len(u)))
#
# match segment_wize:
#
#     case True:
#         harvested_data = self.harvest_state(u, num_heartbeats)
#         filtered_harvest = harvested_data[:, selected_neurons]
#         captured_data = filtered_harvest.T
#
#     case False:
#
#         print("\033[34m Beginning to update reservoir neurons ECG channel pair wize...")
#         for heartbeat in range(num_heartbeats):
#             # this layer will iterate over the different pairs of data points in the input data.
#             for i in range(len(u[heartbeat])):
#                 # this layer will iterate over the different neurons in the reservoir and capture their
#                 # activations.
#                 for j in range(num_neurons):
#                     index = selected_neurons[j]
#                     activation = self.x[index]
#                     captured_data[j][i] = activation
#
#                 # This update is for every pair, the captured data will reflect the update of the state for
#                 # every pair.
#                 sample = np.array(u[heartbeat][i]).T
#                 self.update_state(sample)
#         print("\033[36m Completed capturing state activations.\n")
