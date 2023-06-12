import tensorflow as tf
import numpy as np


# The defining global parameters for the reservoir of the ESN model.
# Nx : The number of neurons in the reservoir.
# Sparsity : The percentage of neurons that are connected to other neurons.
# Distribution pf non-zero weights : The distribution of the non-zero weights in the reservoir.
# Spectral radius : The spectral radius of the reservoir (W).
# The spectral radius is the largest absolute eigenvalue of a matrix.
# The spectral radius is calculated after the reservoir weights W have been generated sparsely.
# Input scaling : The scaling of the input weights (W-in).
# Leaking rate : The leaking rate of the reservoir neurons.


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


def generate_sparse_matrix(n_x, sparsity, lower_bound=-1, upper_bound=1):
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
    # Generate a random list of indices for the non-zero weights in the reservoir.
    # The indices are generated from a uniform distribution.
    # The indices are generated from 0 to n_x - 1.
    # The indices are generated without replacement.
    indices = np.random.choice(np.arange(n_x * n_x), num_non_zero, replace=False)
    # Generate a random list of weights for the non-zero weights in the reservoir.
    # The weights are generated from a uniform distribution.
    # The weights are generated from -1 to 1.
    # The weights are generated from a uniform distribution.
    weights = np.random.uniform(lower_bound, upper_bound, num_non_zero)
    # Assign the weights to the indices in the reservoir.
    w.ravel()[indices] = weights
    # Return the reservoir weights.
    return w


def main():
    """
    Notes on current hyperparameters settings:
    The spectral radius currently when not rescaling the weights is 4.2281, this could be a tad high.
    slightly problematic.
    """
    Nx = 500
    sparseness = 0.1
    little_bound = -1
    big_bound = 1
    #rescale_factor = 0.4

    # Generate the sparse matrix of weights for the reservoir (W).
    sparse_matrix = generate_sparse_matrix(Nx, sparseness, little_bound, big_bound)
    #sparse_matrix *= rescale_factor
    # for row in sparse_matrix: # Looks sparsy to me
    #     print(row)

    # Calculate the spectral radius of the reservoir weights.
    spectral_radius = get_spectral_radius(sparse_matrix)
    print(spectral_radius)
    sparse_matrix /= spectral_radius
    print(get_spectral_radius(sparse_matrix))

    # TODO: go onto the input scaling part of building the reservoir for the ESN model.


if __name__ == '__main__':
    main()
