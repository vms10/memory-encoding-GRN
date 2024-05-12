# -*- coding: utf-8 -*-
"""Tools to simulate reservoir dynamics."""

import warnings

import numpy as np

from utils import get_spectral_radius


def simulate_reservoir_dynamics(weights, input_weights, input_signal,
                                init_state=None, node_function=None, perturbed_node=None):
    """Simulate the dynamics of a reservoir.

    Given the internal weights that define the reservoir, the input weights,
    the input signal stream, the initial state and the integrative function
    that the nodes apply, it computes the dynamics of the reservoir.

    Parameters
    ----------
    weights : numpy array
        Weighted adjacency matrix that defines the connectivity of the
        reservoir. Must be squared.
    input_weights : numpy array
        Weights that determine the effect of the input signal stream on each of
        the nodes of the reservoir. Rows are input streams (there might be more
        than one) and columns correspond to reservoir nodes (equal to number of
        rows and columns of `weights`).
    input_signal : numpy array
        Signal stream, with the value of the input signal for each time step at
        which the dynamics of the network must be computed. Columns are input
        streams and rows timesteps.
    init_state : None, numpy array, optional (default=None)
        Value of the nodes of the reservoir at the initial time step. It must
        be a 1-dimensional vector of the length of the reservoir size. If
        `None` all nodes are initialized at `0`.
    node_function : None, Function, optional (default=None)
        Function applied to the weigted sum of all the incomming connections of
        a node to compute its state. It defaults to the hiperbolic tangent.
    """
    # n: nodes, t: time, c: input chanels
    # input_signal(t x c) dot input_weights(c x n) = input_per_node(t x n)
    # system_state(1 x n) dot weights(n x n) = system_state(1 x n)
    # --> dynamics (t x n)
    nnodes = weights.shape[0]
    input_signal = (input_signal[:, None] if input_signal.ndim == 1
                    else input_signal)
    tsteps, nstreams = input_signal.shape

    if weights.ndim != 2 or nnodes != weights.shape[1]:
        raise RuntimeError("weights should be a squared matrix, but its shape "
                           "is {}".format(weights.shape))

    input_weights = np.atleast_2d(input_weights)
    if (nstreams, nnodes) != input_weights.shape:
        raise RuntimeError("shapes {} and {} are incompatible."
                           .format((nstreams, nnodes), input_weights.shape))

    # Pointers for internal use
    node_function = node_function or np.tanh
    matrix_product = np.dot
    dynamics = np.zeros((tsteps+1, nnodes))
    if init_state is not None:
        dynamics[0, :] = init_state

    input_per_node = matrix_product(input_signal, input_weights)


    # unperturbed trajectory
    for t in range(0, input_signal.shape[0]):
        dynamics[t+1, :] = node_function(
                               matrix_product(dynamics[t, :], weights)
                               + input_per_node[t])
    
    




    return dynamics[1:, :]