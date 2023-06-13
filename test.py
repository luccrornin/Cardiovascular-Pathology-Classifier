import tensorflow as tf
import numpy as np


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


# spectral radius is calculated after the reservoir weights W have been generated sparsely. So we will create a method
# that calculates the spectral radius after the reservoir weights have been generated.
def get_spectral_radius(W):
    """
    This function calculates the spectral radius of the reservoir weights.
    The spectral radius is defined as the largest absolute eigenvalue of a matrix.
    :param W: The reservoir weights.
    :return: The spectral radius of the reservoir weights.
    """
    e = np.linalg.eigvals(W)
    return np.max(np.abs(e))


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


def generate_hidden_matrix(n_x, sparsity, lower_bound=-1, upper_bound=1):
    """
    This function generates a sparse matrix of weights for the reservoir (W).
    :param n_x: The number of neurons in the reservoir.
    :param sparsity: The percentage of neurons that are connected to other neurons.
    :param lower_bound: The lower bound for the uniform distribution the non-zero weights are generated from.
    :param upper_bound: The upper bound for the uniform distribution the non-zero weights are generated from.
    :return: A sparse matrix containing weights for the reservoir (w) of size n_x.
    """
    # Create a matrix of zeros of size n_x.
    w = np.zeros((n_x, n_x))
    # Calculate the number of non-zero weights in the reservoir.
    num_non_zero = int(np.round(sparsity * n_x * n_x))
    print(f'The number of non-zero weights in the reservoir is {num_non_zero:,}')

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
    weights = np.random.uniform(lower_bound, upper_bound, num_non_zero)

    # print(len(indices_x))
    # print(len(indices_y))
    # print(len(weights))
    # Assign the weights to the indices in the reservoir.
    for i in range(num_non_zero):
        x = indices[i][0]
        y = indices[i][1]
        if x > n_x or y > n_x:
            print(f'x is {x} and y is {y}')
        w[x][y] = weights[i]
    # w.ravel()[indices] = weights
    # Return the reservoir weights.
    return w


def generate_sparse_weights(n_x, n_u, sparsity, lower_bound=-1, upper_bound=1):
    """
    This function is responsible for generating the input weights for the ESN model.
    :param n_x: The number of neurons in the reservoir.
    :param n_u: The number of input neurons, corresponding to the number of channels in the input data.
    :param sparsity: The percentage of neurons that are connected to other neurons.
    :param lower_bound: The lower bound for the uniform distribution the non-zero weights are generated from.
    :param upper_bound: The upper bound for the uniform distribution the non-zero weights are generated from.
    :return: A sparse matrix containing weights for the input (w_in) of size n_x by n_u.
    """
    # Create a matrix of zeros of size n_x by n_u.
    w_in = np.zeros((n_x, n_u))

    # Calculate the number of non-zero weights in the reservoir dictated by the sparsity.
    num_non_zero = int(np.round(sparsity * n_x * n_u))

    # Generate a random list of indices for the non-zero weights in the reservoir dictated by the sparsity.
    indices = np.random.choice(np.arange(n_x * n_u), num_non_zero, replace=False)

    # Generate a random list of weights for the non-zero weights in the reservoir.
    weights = np.random.uniform(lower_bound, upper_bound, num_non_zero)

    # Assign the weights to the indices in the reservoir.
    w_in.ravel()[indices] = weights

    # Return the input weights.
    return w_in


def generate_neurons(n_x):
    """
    This function is responsible for generating the initial state of the reservoir.
    :param n_x: The number of neurons in the reservoir.
    :return: The initial state of the reservoir.
    """
    # Generate a random vector of size n_x.
    x = np.random.uniform(-1, 1, n_x)
    # Return the initial state of the reservoir.
    return x


def update_state(x, u, w_in, w, alpha):
    """
    This function is responsible for updating the state x(n) of the reservoir.
    The update of a state involves 2 steps of equations: [;] indicates vertical concatenation.
    First, the update of the state is calculated by the equation: x_update(n) = tanh(w_in*[1;u(n)] + w*x(n-1))
    Second, the state is now calculated by the equation: x(n) = (1-alpha)*x(n-1) + alpha*x_update(n)
    :param x: The state of the reservoir at time n-1.
    :param u: The input data at time n, corresponding to a reading from an EEG channel.
    :param w_in: The input weights of the reservoir.
    :param w: The hidden weights of the reservoir.
    :param alpha: The leak rate of the reservoir.
    :return: The state of the reservoir at time n.
    """
    # Calculate the update of the state.
    x_update = np.tanh(np.dot(w_in, np.vstack((1, u))) + np.dot(w, x))
    # Calculate the state.
    x = (1 - alpha) * x + alpha * x_update  # muahahahahaha i think it's gonnnna work!!!! :D
    # Return the state.
    return x


def train_state(x, u, w_in, w, alpha):
    """
    This function is responsible for training the reservoir of the input data and updating the weights using all the
    traning data.
    :param x: The activations of the reservoir.
    :param u: The input data.
    :param w_in: The input weights of the reservoir.
    :param w: The hidden weights of the reservoir.
    :param alpha: The leak rate of the reservoir.
    :return: The updated activations of the reservoir.
    """
    pass


def main():
    """
    Notes on current hyperparameters settings:
    The spectral radius currently when not rescaling the weights is 4.2281, this could be a tad high.
    slightly problematic.
    """
    Nx = 500
    Nu = 2
    Ny = 10
    alpha = 0.3
    sparseness = 0.1
    little_bound = -1
    big_bound = 1
    # rescale_factor = 0.4

    # create the input weight matrix w_in
    w_in = generate_sparse_weights(Nx, Nu, sparseness, little_bound, big_bound)

    # Generate the sparse matrix of weights for the reservoir (W).
    sparse_matrix = generate_hidden_matrix(Nx, sparseness, little_bound, big_bound)
    # Calculate the spectral radius of the reservoir weights.
    spectral_radius = get_spectral_radius(sparse_matrix)
    # rescale the weights
    w = sparse_matrix / spectral_radius

    # Generate the initial state of the reservoir.
    x = generate_neurons(Nx)

    # Generate the weights for the output layer of the ESN model.
    w_out = generate_sparse_weights(Nx, Ny, sparseness, little_bound, big_bound)

    # Generate the units for the output layer of the ESN model.
    y = generate_neurons(Ny)

    # TODO 1: go onto the input scaling part of building the reservoir for the ESN model.
    # TODO 2: Compare the two methods for generating sparse matrices, and evaluate their spectral radi.
    # TODO 3: Implement the necessary achitecture for the readout layer of the ESN model. (i.e. the output weights)
    # TODO 4: Implement the necessary architecture for the training of the ESN model maybe in a class though.
    # TODO 5: Investigate the reservoir dynamics of the ESN model using the training data.
    # TODO 6: Look into the different methods for training the ESN model.


if __name__ == '__main__':
    main()
