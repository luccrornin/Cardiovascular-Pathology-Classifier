import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import test
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from scipy.signal import butter, filtfilt


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
    heartbeat_types.remove('~')
    heartbeat_types.remove('+')
    for idx in sorted(not_wanted_labels, reverse=True):
        del train_data[idx]
        del label_data[idx]

    return train_data, label_data, heartbeat_types


def re_sample_data(train_data, step, save=False):
    """
    This function will re-sample the data such that every tenth point will be selected from each heartbeat sequence.
    :param train_data: The collection of all heartbeat segments
    :param save: A boolean indicating whether the re-sampled data should be saved to a file.
    :return: The re-sampled data.
    """
    for i, heartbeat in enumerate(train_data):
        altered_heartbeat = heartbeat[::step]
        train_data[i] = np.asarray(altered_heartbeat)

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


def k_fold(data, labels, heartbeat_types):
    n_u = 2
    n_y = 2
    k = 5  # number of folds

    # defining the K-fold cross validation
    kfold = KFold(n_splits=k, shuffle=True)

    # defining the hyperparameters
    hyperparameters = {'n_x': [200, 400, 600],
                       'sparsity': [0.05, 0.1, 0.15, 0.2],
                       'leaking_rate': [0.3, 0.5, 0.1],
                       'lower_bound': [-0.5, -1, -0.75],
                       'upper_bound': [0.5, 1, 0.75],
                       'input_bias': [0.5, 0.75, 1.0]}

    best_score = 0
    best_hyperparameters = None

    # iterating over all combinations of hyperparameters
    for n_x in hyperparameters['n_x']:
        for sparsity in hyperparameters['sparsity']:
            for leaking_rate in hyperparameters['leaking_rate']:
                for bound in range(len(hyperparameters['lower_bound'])):
                    for input_bias in hyperparameters['input_bias']:
                        scores = []
                        # performing k-fold cross validation for each combination of hyperparameters
                        for train, testing in kfold.split(data):
                            esn = test.ESN(n_x, n_u, n_y, input_bias, 1, sparsity, leaking_rate,
                                           hyperparameters['lower_bound'][bound],
                                           hyperparameters['upper_bound'][bound])

                            train_data = [data[i] for i in train]
                            train_labels = np.asarray([labels[i] for i in train])

                            test_data = [data[i] for i in testing]
                            test_labels = np.asarray([labels[i] for i in testing])

                            # splitting the data into training and validation set
                            # train_data, train_labels = data[train], labels[train]
                            # test_data, test_labels = data[testing], labels[testing]

                            # training the ESN
                            esn.train(train_data, train_labels)

                            # testing the ESN
                            score = esn.test(test_data, test_labels, heartbeat_types)

                            scores.append(score)

                        print(
                            f'\033[32m Hyperparameters: Nx:{n_x}\nInput Bias: {input_bias}\nSparness: {sparsity}\nLeaking rate: {leaking_rate}\nlower bound: {hyperparameters["lower_bound"][bound]} & upper bound: {hyperparameters["upper_bound"][bound]}')

                        # computing the average validation score
                        # avg_score = sum(scores) / len(scores)
                        np_scores = np.asarray(scores)
                        avg_score = np.mean(np_scores)
                        variance = np.var(np_scores)
                        # sd = np_scores.std()
                        # score_range = np.max(np_scores) - np.min(np_scores)
                        print(f'\033[32m Average Score: {avg_score}')
                        print(f'\033[32m Variance Score: {variance}')
                        # print(f'\033[32m Standard Deviation: {sd}')
                        # print(f'\033[32m Score Range: {score_range}')

                        # updating the best score and the corresponding hyperparameters
                        if avg_score > best_score:
                            best_score = avg_score
                            best_hyperparameters = (
                                n_x, n_u, n_y, input_bias, sparsity, leaking_rate,
                                hyperparameters['lower_bound'][bound],
                                hyperparameters['upper_bound'][bound])

    print(f'Best Score: {best_score}')
    print(f'Best Hyperparameters: {best_hyperparameters}')


def cross_validation(mv_file, annotation_file):
    Nu = 2
    Ny = 2

    train_data, label_data, heartbeat_types = read_in_csv(mv_file, annotation_file)
    train_data, label_data, heartbeat_types = balance_classes(train_data, label_data, heartbeat_types)
    train_data = re_sample_data(train_data, 2)

    # defining the hyperparameters
    hyperparameters = {
        'Nx': [200, 400],
        'input_bias': [0.5, 1],
        'sparseness': [0.1, 0.15],
        'leaking_rate': [0.3, 0.5],
        'upper_bound': [0.5, 1]
    }
    repetitions = 3
    best_score = 0
    for n_x in hyperparameters['Nx']:
        for input_bias in hyperparameters['input_bias']:
            for sparseness in hyperparameters['sparseness']:
                for leaking_rate in hyperparameters['leaking_rate']:
                    for bound in hyperparameters['upper_bound']:
                        scores = []
                        lower_bound = -1
                        print(f'\033[33mBeginning repetitions')
                        for repetition in range(repetitions):
                            rep_train_data, rep_test_data, rep_train_labels, rep_test_labels = train_test_split(
                                train_data, label_data, test_size=0.6, random_state=42)
                            y_train_target = generate_y_target(rep_train_labels, heartbeat_types)
                            y_test_target = generate_y_target(rep_test_labels, heartbeat_types)

                            esn = test.ESN(n_x, Nu, Ny, input_bias, 1, sparseness, leaking_rate, -bound, bound)

                            # training the ESN
                            esn.train(rep_train_data, y_train_target)

                            # testing the ESN
                            score = esn.test(rep_test_data, y_test_target, heartbeat_types)

                            scores.append(score)

                        # computing the average validation score
                        np_scores = np.asarray(scores)
                        avg_score = np.mean(np_scores)
                        variance = np.var(np_scores)
                        print(f'\033[32m Average Score: {avg_score}')
                        print(f'\033[32m Variance Score: {variance}')
                        print(
                            f'\033[32mHyperparameters: Nx:{n_x}\nInput Bias: {input_bias}\nSparness: {sparseness}\nLeaking rate: {leaking_rate}\nlower bound: {-bound} & upper bound: {bound}')

                        # updating the best score and the corresponding hyperparameters
                        if avg_score > best_score:
                            best_score = avg_score
                            best_hyperparameters = (
                                n_x, Nu, Ny, input_bias, sparseness, leaking_rate, -bound, bound)


def test1(mv_file, annotation_file):
    train_data, label_data, heartbeat_types = read_in_csv(mv_file, annotation_file)
    train_data, label_data, heartbeat_types = balance_classes(train_data, label_data, heartbeat_types)
    resample_step = 2
    train_data = re_sample_data(train_data, resample_step)

    y_target = generate_y_target(label_data, heartbeat_types)

    Nu = 2
    Ny = 2

    Nx = 400
    input_bias = 0.1
    reservoir_bias = 1
    sparseness = 0.1
    little_bound = -1
    big_bound = 1
    alpha = 0.3
    regularisation = 0.2
    washout = 5
    save = True

    # Create the ESN model.
    esn = test.ESN(Nx, Nu, Ny, input_bias, reservoir_bias, sparseness, alpha, little_bound, big_bound)
    esn.set_washout(washout)
    esn.set_regularisation_factor(regularisation)

    harvested_states = esn.harvest_state(train_data, len(train_data))

    esn.train_readout(harvested_states, y_target, save)

    num_neurons_to_plot = 3
    num_neurons_per_plot = 3
    num_heartbeats_to_feed = 1
    state_activation_title = f'Nx = {Nx}, Lbound = {little_bound}, Ubound = {big_bound}'
    # state_activation_title = f'L-bound = {little_bound}, U-bound = {big_bound}'
    segment_wise = False
    esn.timeseries_activation_plot([train_data[0]], num_neurons_to_plot, num_neurons_per_plot, num_heartbeats_to_feed,
                                   state_activation_title, segment_wise)

    # print("The Heartbeat Types are: ", heartbeat_types)
    # esn.set_w_out(np.load('w_out.npy'))
    # debug_output_weights = esn.w_out

    test_beat_1 = train_data[41]
    # test_beat_2 = train_data[1]
    # test_beat_3 = train_data[2]
    # test_beat_4 = train_data[568]
    print(f'\033[32m The class should be {y_target[0]}')
    probabilities = esn.classify(test_beat_1)
    esn.get_y(heartbeat_types)


def main():
    # -------------------------------------------- Data Preprocessing --------------------------------------------------
    mv_file = '106.csv'
    annotation_file = '106annotations - 106annotations.csv'

    # test1(mv_file, annotation_file)
    # test2(mv_file, annotation_file)
    cross_validation(mv_file, annotation_file)

    # train_data, label_data, heartbeat_types = read_in_csv(mv_file, annotation_file)
    # train_data, label_data, heartbeat_types = balance_classes(train_data, label_data, heartbeat_types)
    # resample_step = 2 # experiment with this value and also the washout.
    # # train_data = re_sample_data(train_data, resample_step)
    #
    # train_data, test_data, train_labels, test_labels = train_test_split(train_data, label_data, test_size=0.2, random_state=42)
    #
    # # print(label_data[568])
    #
    # # Generate the target one hot encodings the model should output.
    # y_train_target = generate_y_target(label_data, heartbeat_types)
    # y_test_target = generate_y_target(test_labels, heartbeat_types)
    #
    # k_fold(train_data, y_train_target, heartbeat_types)

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

    # plot_data = count_classe_instances(label_data, heartbeat_types)  # This matrix contains as many lists as there
    # are types of heartbeats. Each list then has every sample of that type.

    # plot_class_distribution(plot_data, heartbeat_types)

    # --------------------------------------------Building the ESN model------------------------------------------------
    # This is the portion of code for .

    Nu = 2
    Ny = 2

    Nx = 400
    input_bias = 0.1
    reservoir_bias = 1
    sparseness = 0.1
    little_bound = -1
    big_bound = 1
    alpha = 0.3
    regularisation = 0.2
    washout = 5
    save = True

    # Create the ESN model.
    # esn = test.ESN(Nx, Nu, Ny, input_bias, reservoir_bias, sparseness, alpha, little_bound, big_bound)
    # esn.set_washout(washout)
    # esn.set_regularisation_factor(regularisation)
    # esn.train(train_data, y_target)
    # esn.test(train_data, y_target)

    # Here we harvest the activations of the reservoir units at the end of each heartbeat sequence.
    # harvested_states = esn.harvest_state(train_data, len(train_data))

    # Here we train the readout weights of the ESN model using the harvested states and the target labels the model
    # should output.
    # esn.train_readout(harvested_states, y_target, save)

    # --------------------------------------------Operating the ESN model-----------------------------------------------


if __name__ == '__main__':
    main()
