import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
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

    def __init__(self, n_x, n_u, n_y, sparsity, leaking_rate, lower_bound=-1, upper_bound=1):
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
        self.sparsity = sparsity
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.alpha = leaking_rate
        placeholder_w = self.scaled_spectral_radius_matrix()
        self.w = placeholder_w[0]
        self.w_non_zero = placeholder_w[1]
        placeholder_w_in = self.generate_sparse_matrix(self.n_x, self.n_u)  # 500 x 2
        self.w_in = placeholder_w_in[0]  # 500 x 2, the third is the bias column full of ones.
        self.w_in_non_zero = placeholder_w_in[1]  # 500 x 2
        placeholder_w_out = self.generate_sparse_matrix(self.n_y, self.n_x)
        self.w_out = placeholder_w_out[0]
        self.w_out_non_zero = placeholder_w_out[1]
        self.x = self.generate_neurons(self.n_x)  # n_x x 1
        self.y = self.generate_neurons(self.n_y)  # n_y x 1
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

        # Generate a random list of indices for the non-zero weights in the reservoir.
        # The indices are generated from a uniform distribution.
        # The indices are generated from 0 to n_x - 1.
        # The indices are generated without replacement.
        # The matrix is 2d, so we need to generate two lists of indices.

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

        # print(len(indices_x))
        # print(len(indices_y))
        # print(len(weights))
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
        x = np.random.uniform(self.lower_bound, self.upper_bound, num_neurons)
        x = np.reshape(x, (num_neurons, 1))  # n_x x 1
        # TODO - Check out if the initilization of the reservoir to zeros is better.
        # x = np.zeros(num_neurons)
        # print(x.shape)
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

    def get_y(self, heartbeat_types):
        """
        This function is responsible for printing the decoded output of the model.
        :return: None
        """
        print(f'\033[32mModel Output: {heartbeat_types[np.argmax(self.y)]}')

    def scaled_spectral_radius_matrix(self):
        """
        This function if responsible for scaling a matrix by its spectral radius.
        """
        # generate a weight matrix for w
        placeholder = self.generate_sparse_matrix(self.n_x, self.n_x)
        w = placeholder[0]
        indices = placeholder[1]
        e = np.linalg.eigvals(w)
        spectral_radius = np.max(np.abs(e))
        # print(f'\033[31m Spectral Radius: {spectral_radius}')
        w /= spectral_radius
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
        input_bias_value = 1
        w_bias_value = 1

        # turn the state activation array into a column vector.
        transpose = self.x.reshape((self.n_x, 1))

        # transpose the input data to be a 2x1 vector.
        u = u.reshape((2, 1))

        u_with_bias = np.vstack((np.array([[input_bias_value]]), u))

        # currently input weight matrix is 500x2, but we need to incorporate the bias term, so we need to make it 500x3
        bias_column = np.ones((self.n_x, 1))
        w_bias_shape = (self.n_x, 1)
        w_bias_col_vec = np.full(w_bias_shape, w_bias_value)
        w_in_with_bias = np.hstack((self.w_in, w_bias_col_vec))
        # result = np.concatenate((u, ones_column), axis=1)
        # w_in_with_bias = np.vstack((self.w_in, bias_column))

        # ---------------------------the update of the state(ESN Guide).------------------------------------------------
        # Calculate the update of the state.
        # print(f'w_in shape should be 500x3: {w_in_with_bias.shape} and the input should be 3x1: {u_with_bias.shape}')
        lhs = np.matmul(w_in_with_bias, u_with_bias)
        rhs = np.matmul(self.w, transpose)
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
        # print(f'\033[34m Calculating readout...')
        reservoir_bias_value = 1
        transpose = self.x.reshape((self.n_x, 1))
        # we now want to insert a bias term into the reservoir state at the 0th index.
        x = np.vstack((np.array([[reservoir_bias_value]]), transpose))
        # print(f'\033[33m X shape: {x.shape}')
        # print(f'\033[33m W_out shape: {self.w_out.shape}')
        # print(f'\033[33m Transpose shape: {transpose.shape}')

        self.y = tf.nn.softmax(np.matmul(self.w_out, x))
        # print(f'\033[36m Readout calculated.\n')
        # print(f'y shape: {self.y.shape}')
        if ret:
            print(f'\033[36m output activations: {self.y}\n')
            return self.y

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

                print("\033[34m Beginning to update reservoir neurons ECG channel pair wize...")
                print(f'\033[33m The length of the captured data is: {len(captured_data)}')

                for heartbeat in range(num_heartbeats):
                    # this layer will iterate over the different pairs of data points in the input data.
                    for sample in range(len(u[heartbeat])):
                        # this layer will iterate over the different neurons in the reservoir and capture their
                        # activations.
                        for neuron in range(num_neurons):
                            index = selected_neurons[neuron]
                            activation = self.x[index][0]
                            # captured_data[neuron][sample] = activation
                            captured_data[neuron].append(activation)

                        # This update is for every pair, the captured data will reflect the update of the state for
                        # every pair.
                        sample = np.array(u[heartbeat][sample]).T
                        self.update_state(sample)
                print("\033[36m Completed capturing state activations.\n")

        state_activation_plot(0, neurons_pp, neurons_pp, num_neurons, captured_data, selected_neurons, title)

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
        harvested_state = []
        print("\033[34m Beginning to harvest reservoir neurons activations...")
        # The outer loop will iterate over the different heartbeats in the input data.
        for heartbeat in range(num_heartbeats):
            # Appending the initial state intuitively didn't seem right, but if needed append here instead.

            # for each heartbeat we want to iterate through the different samples in the heartbeat and update the state.
            for i in range(len(u[heartbeat])):
                # update the state of the reservoir.
                sample = np.array(u[heartbeat][i]).T
                self.update_state(sample)

            # Append the state of the reservoir to the harvested state. Initial state will now be included.
            # The harvest state will only inlclude the activations from the final sample in the heartbeat.
            harvested_state.append(self.x.reshape((self.n_x,)))

        print("\033[36m Completed harvesting reservoir neurons activations.\n")
        return np.asarray(harvested_state)

    def train_state_for_segment(self, u, neuron_index):
        """
        This function is responsible for training the reservoir of the input data and updating the activations
        using all the training data.
        :param u: The input data.
        :return: None
        """
        readout_activations = []
        print("\033[34m Beginning to update reservoir neurons activations for training...")
        # for heartbeat in range(len(u)):
        for i in range(len(u)):
            sample = np.array(u[i]).T

            self.update_state(sample)
            activation = self.get_readout(True)[neuron_index, 0]
            # print(f'Activation: {activation}')
            readout_activations.append(activation)
        print("\033[36m Completed updating reservoir neurons activations for training.\n")

        return readout_activations

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
        """
        reservoir_bias_value = 1
        reservoir_bias_shape = (harvested_states.shape[0], 1)

        # We need to concatenate the bias column to the harvested state activations.
        bias_col = np.full(reservoir_bias_shape, reservoir_bias_value)
        # bias column will be placed as the first column in the harvested states.
        harvested_states = np.concatenate((bias_col, harvested_states), axis=1)

        print("\033[34m Beginning to train the readout weights...")

        self.linear_regression.fit(harvested_states, y_target)

        self.w_out = self.linear_regression.coef_

        if save:
            # we will now save the weights of the matrix, so that we can use them later for other runs.
            np.save('w_out.npy', self.w_out)

        # print(f'\033[33m The shape of the readout weights is: {self.w_out.shape}')

        print("\033[36m Completed training the readout weights.\n")


def generate_random_indices(matrix, n):
    """
    This function generates a random list of indices for a 2D matrix.
    :param matrix: The matrix to generate the indices for.
    :param n: The number of indices to generate.
    :return: The list of indices.
    """
    rows, cols = matrix.shape
    indices = np.random.choice(rows * cols, size=n, replace=False)
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


def state_activation_plot(start, end, step, stop, captured_data, selected_neurons, title):
    """
    This is recursive function that is responsible for plotting the activations of the state activations in the
    reservoir.
    :param start: The index of the first neuron to plot.
    :param end: The index of the last neuron to plot.
    :param step: The number of neurons to plot on the same graph.
    :param stop: The number of neurons to plot in total.
    :param captured_data: The data that was captured from the reservoir to be plotted.
    :param selected_neurons: The neurons that were selected to be plotted.
    :param title: The title of the plot.
    :return:
    """
    # print(f'\033[36m The start is: {start} and the end is: {end}, and the stop is: {stop}')
    for i in range(start, end):
        plt.plot(captured_data[i], label=f'Neuron {selected_neurons[i]}')

    plt.xlabel('Time')
    plt.ylabel('Activation')
    plt.title(title)
    plt.legend()
    plt.show()

    if end + step <= stop:  # this means that we can plot all 4 neurons on the same graph.
        new_start = end
        end = end + step
        state_activation_plot(new_start, end, step, stop, captured_data, selected_neurons, title)
    # the next condition is for when we have less than 4 neurons to plot.
    elif end + step > stop and end != stop:
        new_start = end
        end = stop
        state_activation_plot(new_start, end, step, stop, captured_data, selected_neurons, title)
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
    The spectral radius currently when not rescaling the weights is 4.2281, this could be a tad high.
    slightly problematic.
    """
    Nx = 50
    Nu = 2
    Ny = 10
    sparseness = 0.1
    little_bound = -0.5
    big_bound = 0.5
    alpha = 0.3
    # rescale_factor = 0.4

    # Generate the ESN model.
    esn = ESN(Nx, Nu, Ny, sparseness, alpha, little_bound, big_bound)

    # TODO 1: Investigate the reservoir dynamics of the ESN model using the training data. DONE

    # TODO 2: Create a method to calculate the output weights of the ESN model. DONE

    # TODO 2.1: Linear regression model:
    # TODO 2.1.1: Implement regularisation.
    # TODO 2.1.2: Implement fitting of the model to optimize the output weights. DONE

    # TODO 2.2: Create a ridge regression model to calculate the output weights. yes regularization. But harder.
    # TODO 3: Create a method to carry out k-fold cross validation on the ESN model.

    # TODO potential : go onto the input scaling part of building the reservoir for the ESN model.


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
