import numpy as np
from matplotlib import pyplot as plt

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
        # u = n_u x 1, column vector

    # ---------------------------Initialization methods for the ESN model.----------------------------------------------
    def copilot_sparse_matrix(self, row_dim, col_dim):
        """
        This function is responsible for generating the input weights for the ESN model.
        :param row_dim: The number of neurons in one portion of the reservoir.
        :param col_dim: The number of other neurons in the other portion of the reservoir.
        :return: A sparse matrix containing weights for the input (w_in) of size n_x by n_u.
        """
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
        # Return the initial state of the reservoir.
        return x

    # ---------------------------Helper methods for the ESN model.------------------------------------------------------
    def set_x(self, new_x):
        """
        This function is responsible for setting the state of the reservoir.
        :param new_x: The new state of the reservoir.
        """
        self.x = new_x

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
        w /= spectral_radius
        return w, indices

    # TODO: Investigate this method more to see if it's done correctly, maybe consult with Chat GPT-3.
    def update_state(self, u):
        """
        This function is responsible for updating the state x(n-1) to x(n) of the reservoir .
        The update of a state involves 2 steps of equations: [;] indicates vertical concatenation.
        First, the update of the state is calculated by the equation: x_update(n) = tanh(w_in*[1;u(n)] + w*x(n-1))
        Second, the state is now calculated by the equation: x(n) = (1-alpha)*x(n-1) + alpha*x_update(n)
        :param x: The state of the reservoir at time n-1.
        :param u: The input data at time n, corresponding to a pair of ECG data points. This is given as a 2x1 vector.
        :return: None, the state of the reservoir is updated in place.
        """
        # turn the state activation array into a column vector.
        transpose = self.x.reshape((self.n_x, 1))

        # transpose the input data to be a 2x1 vector.
        u = u.reshape((2, 1))
        u_with_bias = np.vstack((np.array([[1]]), u))

        # currently input weight matrix is 500x2, but we need to incorporate the bias term, so we need to make it 500x3
        bias_column = np.ones((self.n_x, 1))
        w_in_with_bias = np.hstack((self.w_in, bias_column))
        # result = np.concatenate((u, ones_column), axis=1)
        # w_in_with_bias = np.vstack((self.w_in, bias_column))

        # Calculate the update of the state.
        # print(f'w_in shape should be 500x3: {w_in_with_bias.shape} and the input should be 3x1: {u_with_bias.shape}')
        lhs = np.matmul(w_in_with_bias, u_with_bias)
        rhs = np.matmul(self.w, transpose)
        x_update = np.tanh(lhs + rhs)
        # x_update = np.tanh(np.dot(w_in_with_bias, u_with_bias) + np.dot(self.w, transpose))

        # Calculate the state.
        x = (1 - self.alpha) * transpose + self.alpha * x_update  # muahahahahaha i think it's gonnnna work!!!! :D

        if x.shape != (self.n_x, 1):
            raise Exception('The state of the reservoir is not the correct shape.')
        else:
            self.set_x(x)

    # todo: include the bias if needed.
    def get_readout(self):
        """
        This function calculates the readout from the ESN
        The equation for the readout is: y(n) = w_out*[1;x(n)]
        :return: None, the readout is stored in the y attribute.
        """
        print(f'w_out shape: {self.w_out.shape}')
        print(f'x shape: {self.x.shape}')
        transpose = self.x.reshape((self.n_x, 1))
        self.y = np.matmul(self.w_out, transpose)

    def timeseries_activation_plot(self, u, num_neurons):
        """
        This function is responsible for plotting the activations of the state activations in the reservoir.
        The plots we be used to visualize and gain insight into the temporal dynamics of the reservoir.
        Only a subset of the activations will be plotted.
        :param num_neurons: This is the number of neurons to be plotted.
        :param u: The input you want to train the reservoir on. This is the data that will be used to invetiagte the
        temporal dynamics of the reservoir.
        :return: None
        """
        selected_neurons = np.random.choice(np.arange(self.n_x), num_neurons, replace=False)
        captured_data = np.empty((num_neurons, len(u)))

        # this layer will iterate over the different pairs of data points in the input data.
        for i in range(len(u)):
            # this layer will iterate over the different neurons in the reservoir and capture their activations.
            for j in range(num_neurons):
                index = selected_neurons[j]
                activation = self.x[index]
                captured_data[j][i] = activation

            sample = np.array(u[i]).T
            self.update_state(sample)

        state_activation_plot(0, 4, 4, num_neurons, captured_data, selected_neurons)

    # ---------------------------Training the reservoir activation states.----------------------------------------------
    def train_state(self, u):
        """
        This function is responsible for training the reservoir of the input data and updating the weights using all the
        traning data.
        :param u: The input data.
        :return: None
        """
        for i in range(5):
            sample = np.array(u[i]).T
            self.update_state(sample)


# spectral radius is calculated after the reservoir weights W have been generated sparsely. So we will create a method
# that calculates the spectral radius after the reservoir weights have been generated.


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


def state_activation_plot(start, end, step, stop, captured_data, selected_neurons):
    """
    This is recursive function that is responsible for plotting the activations of the state activations in the
    reservoir.
    :param start: The index of the first neuron to plot.
    :param end: The index of the last neuron to plot.
    :param step: The number of neurons to plot on the same graph.
    :param stop: The number of neurons to plot in total.
    :param captured_data: The data that was captured from the reservoir to be plotted.
    :param selected_neurons: The neurons that were selected to be plotted.
    :return:
    """
    print(f'\033[36m The start is: {start} and the end is: {end}, and the stop is: {stop}')

    for i in range(start, end):
        plt.plot(captured_data[i, :], label=f'Neuron {selected_neurons[i]}')

    plt.xlabel('Time')
    plt.ylabel('Activation')
    plt.title('State Activations')
    plt.legend()
    plt.show()

    if end + step <= stop:  # this means that we can plot all 4 neurons on the same graph.
        new_start = end
        end = end + step
        state_activation_plot(new_start, end, step, stop, captured_data, selected_neurons)
    # the next condition is for when we have less than 4 neurons to plot.
    elif end + step > stop and end != stop:
        print("\033[32m for the plot that should only have 3")
        new_start = end
        end = stop
        state_activation_plot(new_start, end, step, stop, captured_data, selected_neurons)
    elif end == stop:
        print("\033[31m I reached the final point!")
        print(f'\033[31m The start is: {start} and the end is: {end}')
        return


def main():
    """
    Notes on current hyperparameters settings:
    The spectral radius currently when not rescaling the weights is 4.2281, this could be a tad high.
    slightly problematic.
    """
    Nx = 500
    Nu = 2
    Ny = 10
    sparseness = 0.1
    little_bound = -0.5
    big_bound = 0.5
    alpha = 0.3
    # rescale_factor = 0.4

    # Generate the ESN model.
    esn = ESN(Nx, Nu, Ny, sparseness, alpha, little_bound, big_bound)

    # TODO 1: Investigate the reservoir dynamics of the ESN model using the training data.
    # TODO 2: Create a method to calculate the output weights of the ESN model.
    # TODO 2.1: Create a linear regression model to calculate the output weights. No regularization. But easiest.
    # TODO 2.2: Create a ridge regression model to calculate the output weights. yes regularization. But harder.
    # TODO 3: Create a method to carry out k-fold cross validation on the ESN model.

    # TODO potential : go onto the input scaling part of building the reservoir for the ESN model.


if __name__ == '__main__':
    main()
