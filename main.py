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
    Scale the input data to a specified output range. All floats
    :param input_data: Input data to be scaled.
    :param input_min: Minimum value of the input data.
    :param input_max: Maximum value of the input data.
    :param output_min: Desired minimum value of the output range.
    :param output_max: Desired maximum value of the output range.
    :return: Scaled input data.
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
                                train_data, label_data, test_size=0.5, random_state=42)
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


def evaluate_model(mv_file, annotations_file):
    """
    This function will evaluate the performance of the model on the test data.
    The model will be tested multiple times on different partitions of the data.
    Each partition will be of a 40% training and 60% testing split.
    :param mv_file: The file containing the mv readings.
    :param annotations_file: The file containing the annotations.
    :return: The average accuracy of the model and the variance.
    """

    train_data, label_data, heartbeat_types = read_in_csv(mv_file, annotations_file)
    train_data, label_data, heartbeat_types = balance_classes(train_data, label_data, heartbeat_types)
    resample_step = 2
    train_data = re_sample_data(train_data, resample_step)

    Nu = 2
    Ny = 2

    Nx = 200
    input_bias = 0.5
    reservoir_bias = 1
    sparseness = 0.15
    leaking_rate = 0.3
    little_bound = -1
    big_bound = 1

    repetitions = 10
    scores = []
    for i in range(repetitions):
        print(f'\033[33m Repetition {i + 1} of {repetitions}')
        esn = test.ESN(Nx, Nu, Ny, input_bias, reservoir_bias, sparseness, leaking_rate, little_bound, big_bound)

        rep_train_data, rep_test_data, rep_train_labels, rep_test_labels = train_test_split(
            train_data, label_data, test_size=0.5, random_state=42)

        y_train_target = generate_y_target(rep_train_labels, heartbeat_types)
        y_test_target = generate_y_target(rep_test_labels, heartbeat_types)

        esn.train(rep_train_data, y_train_target)

        score = esn.test(rep_test_data, y_test_target, heartbeat_types)

        scores.append(score)

    scores = np.array(scores)
    avg_score = np.mean(scores)
    variance = np.var(scores)

    return avg_score, variance


def main():
    # -------------------------------------------- Data Preprocessing --------------------------------------------------
    mv_file = '106.csv'
    annotation_file = '106annotations - 106annotations.csv'

    # test1(mv_file, annotation_file)
    # cross_validation(mv_file, annotation_file)
    # avg_score, variance = evaluate_model(mv_file, annotation_file)
    # print(f'\033[32m The average score is {avg_score} and the variance is {variance}')

    train_data, label_data, heartbeat_types = read_in_csv(mv_file, annotation_file)
    train_data, label_data, heartbeat_types = balance_classes(train_data, label_data, heartbeat_types)
    resample_step = 2 # experiment with this value and also the washout.
    train_data = re_sample_data(train_data, resample_step)

    train_data, test_data, train_labels, test_labels = train_test_split(train_data, label_data, test_size=0.5, random_state=42)


    # Generate the target one hot encodings the model should output.
    y_train_target = generate_y_target(train_labels, heartbeat_types)
    y_test_target = generate_y_target(test_labels, heartbeat_types)

    # --------------------------------------------Building the ESN model------------------------------------------------
    # This is the portion of code for .

    Nu = 2
    Ny = 2

    Nx = 200
    input_bias = 0.5
    reservoir_bias = 1
    sparseness = 0.15
    little_bound = -1
    big_bound = 1
    alpha = 0.3
    regularisation = 0.2
    washout = 5
    save = True

    # Create the ESN model.

    esn = test.ESN(Nx, Nu, Ny, input_bias, reservoir_bias, sparseness, alpha, little_bound, big_bound)
    esn.train(train_data, y_train_target)
    esn.test(test_data, y_test_target, heartbeat_types)
    # if you want to classify a single beat then use the classify function and pass in the beat.


if __name__ == '__main__':
    main()
