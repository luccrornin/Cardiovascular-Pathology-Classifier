import pickle
import random
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import tensorflow as tf

"""
The defining global parameters for the reservoir of the ESN model.
Nx : The number of neurons in the reservoir.
Sparsity : The percentage of neurons that are connected to other neurons.
Distribution pf non-zero weights : The distribution of the non-zero weights in the reservoir.
Spectral radius : The spectral radius of the reservoir (W).
The spectral radius is the largest absolute eigenvalue of a matrix.
The spectral radius is calculated after the reservoir weights W have been generated sparsely.
Input scaling : The scaling of the input weights (W-in).
Leaking rate : The leaking rate of the reservoir neurons.
"""


class ESN:
    """
    This class is responsible for creating the Echo State Network (ESN) model.
    """

    def __init__(self, n_x, n_u, n_y, input_bias, reservoir_bias, sparsity, leaking_rate, lower_bound=-1,
                 upper_bound=1):
        """
        This function is responsible for initializing the ESN model.
        :param n_x: The number of neurons in the hidden part of the reservoir.
        :param n_u: The number of neurons in the input part of the reservoir.
        :param n_y: The number of neurons in the output part of the reservoir.
        :param sparsity: The percentage of neurons that are connected to other neurons. Kept constant for all
        weight matrices.
        :param leaking_rate: The leaking rate of the reservoir neurons.
        :param lower_bound: The lower bound for the uniform distribution the non-zero weights are generated from.(const)
        :param upper_bound: The upper bound for the uniform distribution the non-zero weights are generated from.(const)
        """
        self.n_x = n_x
        self.n_u = n_u
        self.n_y = n_y
        self.alpha = leaking_rate
        self.sparsity = sparsity
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.input_bias = input_bias
        self.reservoir_bias = reservoir_bias
        self.washout = 3  # The washout here is the number of time the heartbeat will be fed into the reservoir to
        # ensure a washout of the initial state of the reservoir.
        self.regularisation_factor = 0.6
        placeholder_w = self.scaled_spectral_radius_matrix(-0.2)
        self.w = placeholder_w[0]
        self.w_non_zero = placeholder_w[1]
        placeholder_w_in = self.generate_sparse_matrix(self.n_x, self.n_u)  # 500 x 2
        self.w_in = placeholder_w_in[0]  # 500 x 2, the third is the bias column full of ones.
        self.w_in_non_zero = placeholder_w_in[1]  # 500 x 2
        placeholder_w_out = self.generate_sparse_matrix(self.n_y, self.n_x)
        self.w_out = placeholder_w_out[0]
        self.w_out_non_zero = placeholder_w_out[1]
        self.x = self.generate_neurons(self.n_x)  # n_x x 1
        # self.y = self.generate_neurons(self.n_y)  # n_y x 1
        self.y = None
        self.linear_regression = LinearRegression()
        # u = n_u x 1, column vector

    # ---------------------------Initialization methods for the ESN model.----------------------------------------------

    def generate_sparse_matrix(self, row_dim, col_dim):
        """
        This function generates a sparse matrix of weights for the reservoir (W).
        :param row_dim: The number of neurons in one portion of the reservoir.
        :param col_dim: The number of other neurons in the other portion of the reservoir.
        :return: A sparse numpy matrix containing weights for the reservoir (w) of size n_x and the indices of the
        non-zero weights in the reservoir.
        """
        # Create a matrix of zeros of size n_x.
        w = np.zeros((row_dim, col_dim))
        # Calculate the number of non-zero weights in the reservoir.
        num_non_zero = int(np.round(self.sparsity * row_dim * col_dim))

        # we need to generate two lists of indices, one for the x-axis and one for the y-axis.
        # The indices are generated from 0 to n_x - 1.
        # indices_x = np.random.choice(np.arange(n_x*n_x), num_non_zero, replace=False)
        # indices_y = np.random.choice(np.arange(n_x*n_x), num_non_zero, replace=False)
        indices = generate_random_indices(w, num_non_zero)

        # Generate a random list of weights for the non-zero weights in the reservoir.
        # The weights are generated from a uniform distribution.
        # The weights are generated from -1 to 1.
        # The weights are generated from a uniform distribution.
        weights = np.random.uniform(self.lower_bound, self.upper_bound, num_non_zero)

        # Assign the weights to the indices in the reservoir.
        for i in range(num_non_zero):
            x = indices[i][0]
            y = indices[i][1]
            if x > row_dim or y > col_dim:
                raise Exception('The indices are out of bounds.')
            w[x][y] = weights[i]
        # w.ravel()[indices] = weights
        # Return the reservoir weights.
        return w, indices

    def generate_neurons(self, num_neurons):
        """
        This function is responsible for generating the initial state of the reservoir.
        :param num_neurons: The number of neurons to be generated for a layer.
        :return: The initial state of the reservoir.
        """

        # Generate a random vector of size n_x.
        # x = np.random.uniform(self.lower_bound, self.upper_bound, num_neurons)
        # x = np.reshape(x, (num_neurons, 1))  # n_x x 1
        x = np.zeros(num_neurons)
        x = np.reshape(x, (num_neurons, 1))  # n_x x 1

        # Return the initial state of the reservoir.
        return x

    # ---------------------------Helper methods for the ESN model.------------------------------------------------------
    def set_x(self, new_x):
        """
        This function is responsible for setting the state of the reservoir.
        :param new_x: The new state of the reservoir.
        """
        self.x = new_x

    def set_w_out(self, new_w_out):
        """
        This function is responsible for setting the output weight matrix to a specified or pre-trained matrix.
        :param new_w_out: The new output weight matrix.
        :return: None
        """
        self.w_out = new_w_out

    def set_washout(self, new_washout):
        """
        This function is responsible for setting the washout of the reservoir.
        The washout here is how many times a single heartbeat segment will be fed into the reservoir to washout the
        initial state of the reservoir.
        :param new_washout: The new washout of the reservoir.
        :return: None, simply update the attribute of the class.
        """
        self.washout = new_washout

    def set_regularisation_factor(self, new_regularisation_factor):
        """
        This function is responsible for setting the regularisation factor of the model.
        :param new_regularisation_factor: The new regularisation factor of the model.
        :return: None, simply update the attribute of the class.
        """
        self.regularisation_factor = new_regularisation_factor

    def set_linear_model(self, new_linear_model):
        """
        This function is responsible for setting the linear model to a specified or pre-trained model.
        :param new_linear_model: The file name of the new linear model.
        :return: None, simply sets the attribute of the class.
        """
        with open(new_linear_model, 'rb') as file:
            loaded_model = pickle.load(file)

        self.linear_regression = loaded_model

    def get_y(self, heartbeat_types):
        """
        This function is responsible for printing the decoded output of the model.
        :return: None
        """
        print(f'\033[32mModel Output: {heartbeat_types[np.argmax(self.y)]}')

    def scaled_spectral_radius_matrix(self, offset=0.0):
        """
        This function if responsible for scaling a matrix by its spectral radius.
        :param offset: The offset to be added to the spectral radius. 0.2 would result in a spectral radius of 0.8.
        """
        # generate a weight matrix for w
        placeholder = self.generate_sparse_matrix(self.n_x, self.n_x)
        w = placeholder[0]
        indices = placeholder[1]
        e = np.linalg.eigvals(w)
        spectral_radius = np.max(np.abs(e))
        # print(f'\033[31m Spectral Radius od the model: {spectral_radius}')
        spectral_radius += spectral_radius * offset
        w /= spectral_radius

        # print(f'\033[31m The new spectral radius of the model: {np.max(np.abs(np.linalg.eigvals(w)))}')
        return w, indices

    def update_state(self, u):
        """
        This function is responsible for updating the state x(n-1) to x(n) of the reservoir .
        The update of a state involves 2 steps of equations: [;] indicates vertical concatenation.
        First, the update of the state is calculated by the equation: x_update(n) = tanh(w_in*[1;u(n)] + w*x(n-1))
        Second, the state is now calculated by the equation: x(n) = (1-alpha)*x(n-1) + alpha*x_update(n)
        :param u: The input data at time n, corresponding to a pair of ECG data points. This is given as a 2x1 vector.
        :return: None, the state of the reservoir is updated in place.
        """

        # turn the state activation array into a column vector.
        transpose = self.x.reshape((self.n_x, 1))

        # transpose the input data to be a 2x1 vector.
        u = u.reshape((2, 1))

        u_with_bias = np.vstack((np.array([[self.input_bias]]), u))

        # currently input weight matrix is 500x2, but we need to incorporate the bias term, so we need to make it 500x3
        bias_column = np.ones((self.n_x, 1))
        w_bias_shape = (self.n_x, 1)
        w_bias_col_vec = np.full(w_bias_shape, self.reservoir_bias)
        w_in_with_bias = np.hstack((self.w_in, w_bias_col_vec))
        # result = np.concatenate((u, ones_column), axis=1)
        # w_in_with_bias = np.vstack((self.w_in, bias_column))

        # ---------------------------the update of the state(ESN Guide).------------------------------------------------
        # Calculate the update of the state.
        lhs = np.dot(w_in_with_bias, u_with_bias)
        rhs = np.dot(self.w, transpose)
        x_update = np.tanh(lhs + rhs)
        # x_update = np.tanh(np.dot(w_in_with_bias, u_with_bias) + np.dot(self.w, transpose))

        # Calculate the state.
        x = (1 - self.alpha) * transpose + self.alpha * x_update

        # ---------------------------the update of the state(LC notes)--------------------------------------------------
        # lhs = np.matmul(self.w, self.x)
        # lhs = lhs.reshape((self.n_x, 1))
        # rhs = np.matmul(self.w_in, u)
        # x = np.tanh(lhs + rhs + 1)

        if x.shape != (self.n_x, 1):
            raise Exception(f'The state of the reservoir is not the correct shape. {self.x.shape}')
        else:
            self.set_x(x)

    def get_readout(self, ret=False):
        """
        This function calculates the readout from the ESN
        The equation for the readout is: y(n) = w_out*[1;x(n)]
        :param ret: A boolean value indicating whether to return the readout.
        :return: None, the readout is stored in the y attribute.
        """
        transpose = self.x.reshape((self.n_x, 1))

        # we now want to insert a bias term into the reservoir state at the 0th index.
        x = np.vstack((np.array([[self.reservoir_bias]]), transpose))

        # Some check for the shape of the readout weight matrix and the state of the reservoir.
        if x.shape != (self.n_x + 1, 1):
            raise Exception(f'The state of the reservoir is not the correct shape. {self.x.shape}')
        elif self.w_out.shape != (self.n_y, self.n_x + 1):
            raise Exception(f'The readout weight matrix is not the correct shape. {self.w_out.shape}')

        # Don't know which activation function to use for the classification task.

        # self.y = get_softmax_probs(np.dot(self.w_out, x))
        # print(f'\033[33m output before relu: {np.dot(self.w_out, x)}')
        self.y = tf.sigmoid(np.dot(self.w_out, x))
        # self.y = tf.nn.relu(np.dot(self.w_out, x))
        # self.y = tf.nn.softmax(np.dot(self.w_out, x))
        # print(f'\033[33m output after relu: {self.y}')
        self.y = self.y.numpy()
        self.y = np.reshape(self.y, (self.n_y,))
        # print(f'\033[33m The readout is: {np.reshape(self.y, (self.n_y,))}')
        # print(f'\033[33m The readout is: {self.y.T.shape}')

        # output = np.tanh(np.dot(self.w_out, x))

        if ret:
            return self.y.T

    def timeseries_activation_plot(self, u, num_neurons, neurons_pp, num_heartbeats, title, segment_wize=True):
        """
        This function is responsible for plotting the activations of the state activations in the reservoir.
        The plots we be used to visualize and gain insight into the temporal dynamics of the reservoir.
        Only a subset of the activations will be plotted.
        :param u: The input you want to train the reservoir on. This is the data that will be used to invetiagte the
        temporal dynamics of the reservoir.
        :param num_neurons: This is the number of neurons to be plotted.
        :param neurons_pp: The number of neurons to be plotted on each plot.
        :param num_heartbeats: This is the number of heartbeats to be fed into the reservoir, the more heartbeats the
        longer the plot will be.
        :param title: The title of the plot.
        :param segment_wize: Boolean indicating whether the plot should show updates for Heartbeat segment to segment,
        or if it should plot the changes from the pair channels in the heartbeat sequences.
        :return: None
        """
        selected_neurons = np.random.choice(np.arange(self.n_x), num_neurons, replace=False)
        # captured_data = np.empty((num_neurons, len(u)))
        # Create a list of lists to store the captured data. The number of inner lists is equal to the number of
        # neurons to be plotted.
        captured_data = [[] for _ in range(num_neurons)]

        match segment_wize:

            case True:
                harvested_data = self.harvest_state(u, num_heartbeats)
                filtered_harvest = harvested_data[:, selected_neurons]
                captured_data = filtered_harvest.T

            case False:

                print("\033[34m Beginning to update reservoir neurons ECG channel pair wize for plotting...")

                # captured_data = [[[] for _ in range(num_neurons)] for _ in range(num_heartbeats)]
                captured_data = []

                # this layer will iterate over the different heartbeats in the training data.
                for heartbeat in range(num_heartbeats):
                    activations = self.train_state_for_segment(u[heartbeat], self.washout, True)
                    filtered_activations = activations[:, selected_neurons]
                    # debug_filtered = np.asarray(filtered_activations.T)
                    captured_data.append(filtered_activations.T)

                # for heartbeat in range(num_heartbeats):
                #     # this layer will iterate over the different pairs of data points in the input data.
                #     for sample in range(len(u[heartbeat])):
                #
                #         # this layer will iterate over the different neurons in the reservoir and capture their
                #         # activations.
                #         for neuron in range(num_neurons):
                #             index = selected_neurons[neuron]
                #             activation = self.x[index][0]
                #             # captured_data[neuron][sample] = activation
                #             captured_data[neuron].append(activation)
                #
                #         # This update is for every pair, the captured data will reflect the update of the state for
                #         # every pair.
                #         sample = np.array(u[heartbeat][sample]).T
                #         self.update_state(sample)
                print("\033[36m Completed capturing state activations for plotting.\n")
        for heartbeat in range(num_heartbeats):
            input_heartbeat = np.asarray(u[heartbeat]).T
            state_activation_plot(0, neurons_pp, neurons_pp, num_neurons, captured_data[heartbeat], input_heartbeat,
                                  selected_neurons, title)

    # ---------------------------Training the reservoir activation states.----------------------------------------------

    def harvest_state(self, u, num_heartbeats):
        """
        This function is responsible for harvesting the activations of the state of the reservoir.
        The input will be fed into the network and the activations of the state will for that sample will be stored.
        This will then be repeated for all the samples in the input data. Yielding a matrix of size (num_samples, n_x).
        Each column representing the progression of a neuron throughout time.
        :param u: The input data.
        :param num_heartbeats: The number of heartbeats to use in the training data.
        :return: The state activations of the reservoir as a numpy array.
        """
        harvested_states = []
        # print("\033[34m Beginning to harvest activations of neurons in the reservoir...")
        # The outer loop will iterate over the different heartbeats in the input data.
        for heartbeat in range(num_heartbeats):
            # reset the state of the reservoir, before feeding in the next heartbeat.
            self.set_x(self.generate_neurons(self.n_x))
            self.train_state_for_segment(u[heartbeat], self.washout)
            harvested_states.append(self.x.reshape((self.n_x,)))

            # we now want to set an outer loop to run the heartbeats through the reservoir multiple time to washout the
            # initial state of the reservoir.
            # for training_sequence in range(heartbeat_repition):
            #     # for each heartbeat we want to iterate through the different samples in the heartbeat and update the
            #     # state.
            #     for i in range(len(u[heartbeat])):
            #         # update the state of the reservoir.
            #         sample = np.array(u[heartbeat][i]).T
            #         self.update_state(sample)
            #
            #     # We want to collect the final state of the reservoir activations, due to the multiple cycles of
            #     # updated for a singular heartbeat, the initial state of the reservoir will be washed out.
            #     if training_sequence == heartbeat_repition - 1:
            #         harvested_state.append(self.x.reshape((self.n_x,)))

        # print(f'\033[33m The shape of the harvested state is: {np.asarray(harvested_states).shape}')
        # print("\033[36m Completed harvesting reservoir neuron activations.\n")
        return np.asarray(harvested_states)

    def train_state_for_segment(self, u, washout, ret=False):
        """
        This function is responsible for training the reservoir of the input data and updating the activations
        using all the training data.
        :param u: The input data.
        :param washout: The number of times to drive the reservoir with the same heartbeat to ensure the initial state
        of the reservoir is washed out.
        :param ret: Boolean indicating whether the function should return the final activations of the reservoir.
        :return: The final activations of the reservoir, as a numpy array. If ret is False, otherwise None.
        """
        activations = []

        # We don't want to inlcude the initial states of the reservoir, as these are not representative of the
        # activations of the reservoir. We want to washout the initial state of the reservoir, hence we will drive the
        # reservoir with the same heartbeat for a number of times.
        for training_sequence in range(washout):
            for i in range(len(u)):
                # change the shape of the input data to be a column vector, as this is what is needed for the update.
                sample = np.array(u[i]).T

                self.update_state(sample)

                # We now want to collect the final state of the reservoir activations.
                if ret and training_sequence == washout - 1:
                    activations.append(self.x.reshape((self.n_x,)))

        # activations.append(self.x.reshape((self.n_x,)))

        # print("\033[36m Completed updating reservoir neurons activations for the heartbeat sequence.\n")
        if ret:
            # print(f'\033[33m The shape of the activations is: {np.asarray(activations).shape}')
            return np.asarray(activations)

    def train_readout(self, harvested_states, y_target, save=False):
        """
        This function is responsible for training the readout weights of the ESN.
        The output weight matrix resulting from the linear regression will be stored in the w_out attribute and also
        saved to an .npy file for later use.
        :param harvested_states: The state activations of the reservoir, for each sample in the training data,
        (num_samples, n_x).
        :param y_target: The target output data being a matrix of one hot encodings. (num_samples, num_classes)
        :param save: Boolean indicating whether the readout weights should be saved to a file.
        :return: None
        Note: The saved version of the readout weights has the bias included, rather the initial matrix does not.
        """
        # The bias is a column vector, so here we simply create the shape dimensions for the bias column.
        reservoir_bias_shape = (harvested_states.shape[0], 1)

        # We will not create the bias column vector, which will be concatenated to the harvested states.
        bias_col = np.full(reservoir_bias_shape, self.reservoir_bias)

        # bias column will be placed as the first column in the harvested states.
        harvested_states = np.concatenate((bias_col, harvested_states), axis=1)

        # print("\033[34m Beginning to train the readout weights...")

        self.w_out = ridge_regression(harvested_states, y_target, self.regularisation_factor)

        # print(f'\033[33m The shape of the readout weights is: {self.w_out.shape}')
        if save:
            # we will now save the weights of the matrix, so that we can use them later for other runs.
            np.save('w_out.npy', self.w_out)

        # print(f'\033[33m The shape of the readout weights is: {self.w_out.shape}')

        # print("\033[36m Completed training the readout weights.\n")

    def classify(self, u):
        """
        This function is responsible for classifying the given heartbeat segment.
        :param u: The heartbeat segment to be classified.
        :return: The predicted class.
        """
        # First we need to get the activations of the states once it has completed processing the heartbeat segment.
        washout = 3  # This is the number of times we will drive the reservoir with the same heartbeat.
        self.train_state_for_segment(u, washout)
        self.get_readout()

        return self.y

    def train(self, train_data, train_labels):
        """
        This method is responsible for training the ESN model from training data and getting is ready for classifying.
        :param train_data: The training data consisting of lists of heartbeat segments.
        :param train_labels: The one hot encodings which correspond to the training data.
        :return: None
        """
        harvested_states = self.harvest_state(train_data, len(train_data))
        self.train_readout(harvested_states, train_labels, save=True)

    def test(self, test_data, test_labels, heartbeat_types):
        """
        This function is responsible for evaluating the performance of the ESN on the test or validation data.
        :param test_data: Your test or validation data
        :param test_labels: The one hot encodings which correspond to the test data.
        :return: Returns the accuracy of the ESN on the test data.
        """
        # First we will get all out predictions for the test data.
        predictions = []
        for heartbeat in test_data:
            heartbeat_prediction = self.classify(heartbeat)
            predictions.append(heartbeat_prediction)

        # Now with our predictions we will evaluate the performance of the ESN.
        # The following 4 lines calculate the accuracy of the model.
        # predictions = np.asarray(predictions)
        # predicted_classes = np.argmax(predictions, axis=1)
        # true_classes = np.argmax(test_labels, axis=1)
        # accuracy = np.sum(predicted_classes == true_classes) / len(true_classes)
        accuracy = calculate_accuracy(np.array(predictions), test_labels, heartbeat_types)

        print(f'\033[32m The accuracy of the ESN on the test data is: {accuracy}')
        return accuracy


def calculate_accuracy(predictions, true_classes, heartbeat_types):
    """
    This function is repsonsible for calculating the accuracy of the ESN on the test data.
    :param predictions: The predictions made by the ESN on the test data.
    :param true_classes: The true classes of the test data.
    :param heartbeat_types: The different types of heartbeats.
    :return: The accuracy of the ESN on the test data.
    """
    if len(predictions) != len(true_classes):
        raise ValueError("The number of predictions and true classes must be equal.")
    num_correct = 0
    for i in range(len(predictions)):
        # make an array of zeros, and set the index of the maximum value to 1.
        prediction = heartbeat_types[np.argmax(predictions[i])]
        true_class = heartbeat_types[np.argmax(true_classes[i])]
        if prediction == true_class:
            num_correct += 1

    return (num_correct / len(predictions)) * 100


def get_softmax_probs(output_matrix):
    # output_matrix -= np.max(output_matrix, axis=1, keepdims=True)
    exp_output = np.exp(output_matrix)
    return exp_output / np.sum(exp_output, axis=1, keepdims=True)


def ridge_regression(x, y, reg):
    """
    This function performs ridge regression on the given data.
    :param x: The harvested states of the reservoir. (num_samples, n_x)
    :param y: The target output data. (num_samples, num_classes)
    :param reg: The regularisation parameter.
    :return:
    """
    x = x.T
    y = y.T

    return np.matmul(np.matmul(np.linalg.inv(np.matmul(x, x.T) + reg * np.identity(x.shape[0])), x), y.T).T


def generate_random_indices(matrix, n):
    """
    This function generates a random list of indices for a 2D matrix.
    :param matrix: The matrix to generate the indices for.
    :param n: The number of indices to generate.
    :return: The list of indices.
    """
    rows, cols = matrix.shape
    indices = np.random.choice(rows * cols, size=n, replace=False)
    # indices = random.sample(matrix, n)
    row_indices = indices // cols
    col_indices = indices % cols
    return list(zip(row_indices, col_indices))


def print_matrix(matrix):
    """
    This function is responsible for printing a matrix.
    :param matrix: The matrix to be printed.
    """
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            print(matrix[i][j], end=' ')
            if matrix[i][j] > 1:
                raise Exception('The matrix is not sparse.')
        print('\n\n')


def state_activation_plot(start, end, step, stop, captured_data, input_heartbeat, selected_neurons, title):
    """
    This is recursive function that is responsible for plotting the activations of the state activations in the
    reservoir.
    The number of neurons to plot per graph can't be less than the number of neurons to plot in total.
    :param start: The index of the first neuron to plot.
    :param end: The index of the last neuron to plot.
    :param step: The number of neurons to plot on the same graph.
    :param stop: The number of neurons to plot in total.
    :param captured_data: The data that was captured from the reservoir to be plotted.
    :param input_heartbeat: The input heartbeat that was fed into the reservoir.
    :param selected_neurons: The neurons that were selected to be plotted.
    :param title: The title of the plot.
    :return:
    """
    # print(f'\033[36m The start is: {start} and the end is: {end}, and the stop is: {stop}')
    for i in range(start, end):
        plt.plot(captured_data[i], label=f'Neuron {selected_neurons[i]}')

    plt.plot(input_heartbeat[0], linestyle='--', label='MlII')
    plt.plot(input_heartbeat[1], linestyle='--', label='V1')

    plt.plot()
    plt.xlabel('Samples')
    plt.ylabel('Activation')
    plt.title(title)
    plt.legend()
    plt.show()

    if end + step <= stop:  # this means that we can plot all 4 neurons on the same graph.
        new_start = end
        end = end + step
        state_activation_plot(new_start, end, step, stop, captured_data, input_heartbeat, selected_neurons, title)
    # the next condition is for when we have less than 4 neurons to plot.
    elif end + step > stop and end != stop:
        new_start = end
        end = stop
        state_activation_plot(new_start, end, step, stop, captured_data, input_heartbeat, selected_neurons, title)
    elif end == stop:
        print('\033[36m Completed plotting the state activations.')
        return


def output_activation_plot(esn, neuron_index, u):
    """
    This function is responsible for plotting the output activations of a specific neuron.
    :return:
    """
    activation = np.zeros((len(u) + 1))
    # When we run the following function, it runs through a heartbeat sequence, updates the states, and then calculates
    # the output activations. The function only returns the activation for the neuron selected.
    readout_activations = esn.train_state_for_segment(u, neuron_index)
    readout_activations.append(esn.get_readout(True)[neuron_index, 0])
    # Note: The output activation seems to be one constantly, which is not what we want. I am not sure as to why though.
    plt.plot(readout_activations, label=f'Neuron {neuron_index}')
    plt.show()


def plot_input_data(u, title="Input Data"):
    """
    This function is responsible for plotting the input data.
    :param u: The input data. (num_samples, num_channels)
    :param title: The title of the plot.
    :return: None, just plots the data.
    """

    channel_1 = [u[i][0] for i in range(len(u))]
    channel_2 = [u[i][1] for i in range(len(u))]

    # channel_1 = [u[i][j][0] for i in range(len(u)) for j in range(len(u[i]))]
    # channel_2 = [u[i][j][1] for i in range(len(u)) for j in range(len(u[i]))]

    # print(channel_1)

    plt.plot(channel_1, label='MLII')
    plt.plot(channel_2, label='V1')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('ECG Mv reading')
    plt.title(title)
    plt.show()


def main():
    """
    Notes on current hyperparameters settings:
    Spectral radius:
    Sparseness:
    alpha:
    Input scaling:
    Reservoir size:
    Bounds for generating the reservoir weights(W, W_in):
    State(x) bias:
    Input bias:

    """
    Nx = 50
    Nu = 2
    Ny = 7
    sparseness = 0.1
    little_bound = -1
    big_bound = 1
    alpha = 0.3
    # rescale_factor = 0.4

    # Generate the ESN model.
    esn = ESN(Nx, Nu, Ny, sparseness, alpha, little_bound, big_bound)

    # TODO: Investigate the weights of the ouput weight matrix as they can get very large.
    # TODO: Implement the ridge regression function that's there for the output weights.
    # TODO: Implement cross validation to find the best hyperparameters.


if __name__ == '__main__':
    main()

"""
    def copilot_sparse_matrix(self, row_dim, col_dim):

        This function is responsible for generating the input weights for the ESN model.
        :param row_dim: The number of neurons in one portion of the reservoir.
        :param col_dim: The number of other neurons in the other portion of the reservoir.
        :return: A sparse matrix containing weights for the input (w_in) of size n_x by n_u.

        # Create a matrix of zeros of size row_dim by col_dim.
        sparse_matrix = np.zeros((row_dim, col_dim))

        # Calculate the number of non-zero weights in the reservoir dictated by the sparsity.
        num_non_zero = int(np.round(self.sparsity * row_dim * col_dim))

        # Generate a random list of indices for the non-zero weights in the reservoir dictated by the sparsity.
        indices = np.random.choice(np.arange(row_dim * col_dim), num_non_zero, replace=False)

        # Generate a random list of weights for the non-zero weights in the reservoir.
        weights = np.random.uniform(self.lower_bound, self.upper_bound, num_non_zero)

        # Assign the weights to the indices in the reservoir.
        sparse_matrix.ravel()[indices] = weights

        # Return the input weights.
        return sparse_matrix
"""
