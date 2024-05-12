# -*- coding: utf-8 -*-
"""Datasets used to test the performance of Reservoir Computing setups."""

import functools
import warnings

import numpy as np


def keep_bounded(dataset_func, max_trials=100, threshold=1e5):
    """Wrapper function to regenerate datasets when they get unstable."""
    @functools.wraps(dataset_func)
    def stable_dataset(n_samples=10, sample_len=1000):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "overflow", RuntimeWarning)
            for i in range(max_trials):
                [x, y] = dataset_func(n_samples=n_samples,
                                      sample_len=sample_len)
                if np.max((x, y)) < threshold:
                    return [x, y]
            else:
                errMsg = ("It was not possible to generate dataseries with {} "
                          "bounded by {} in {} trials.".format(
                              dataset_func.__name__, threshold, max_trials))
                raise RuntimeError(errMsg)

    return stable_dataset


@keep_bounded
def narma10(n_samples=10, sample_len=1000):
    """
    Return data for the 10th order NARMA task.

    Generate a dataset with the 10th order Non-linear AutoRegressive Moving
    Average.

    Parameters
    ----------
    n_samples : int, optional (default=10)
        number of example timeseries to be generated.
    sample_len : int, optional (default=1000)
        length of the time-series in timesteps.

    Returns
    -------
    inputs : list (len `n_samples`) of arrays (shape `(sample_len, 1)`)
        Random input used for each sample in the dataset.
    outputs : list (len `n_samples`) of arrays (shape `(sample_len, 1)`)
        Output of the 30th order NARMA dataset for the input used.

    WARNING: this is an unstable dataset. There is a small chance the system
    becomes unstable, leading to an unusable dataset. It is better to use
    NARMA30 which where this problem happens less often.
    """
    system_order = 10
    inputs, outputs = [], []
    for sample in range(n_samples):
        inputs.append(np.random.rand(sample_len, 1) * .5)
        inputs[sample].shape = (-1, 1)
        outputs.append(np.zeros((sample_len, 1)))
        for k in range(system_order-1, sample_len - 1):
            outputs[sample][k + 1] = .3 * outputs[sample][k] +         \
                .05 * outputs[sample][k] *                             \
                np.sum(outputs[sample][k - (system_order-1):k+1]) +    \
                1.5 * inputs[sample][k - 9] * inputs[sample][k] + .1
    return inputs, outputs


@keep_bounded
def narma30(n_samples=10, sample_len=1000):
    """
    Return data for the 30th order NARMA task.

    Generate a dataset with the 30th order Non-linear AutoRegressive Moving
    Average.

    Parameters
    ----------
    n_samples : int, optional (default=10)
        number of example timeseries to be generated.
    sample_len : int, optional (default=1000)
        length of the time-series in timesteps.

    Returns
    -------
    inputs : list (len `n_samples`) of arrays (shape `(sample_len, 1)`)
        Random input used for each sample in the dataset.
    outputs : list (len `n_samples`) of arrays (shape `(sample_len, 1)`)
        Output of the 30th order NARMA dataset for the input used.
    """
    system_order = 30
    inputs, outputs = [], []
    for sample in range(n_samples):
        inputs.append(np.random.rand(sample_len, 1) * .5)
        inputs[sample].shape = (-1, 1)
        outputs.append(np.zeros((sample_len, 1)))
        for k in range(system_order-1, sample_len - 1):
            outputs[sample][k + 1] = .2 * outputs[sample][k] +          \
                .04 * outputs[sample][k] *                              \
                np.sum(outputs[sample][k - (system_order-1):k+1]) +     \
                1.5 * inputs[sample][k - 29] * inputs[sample][k] + .001
    return inputs, outputs


from operator import xor
def xor_func( p = 0.2, batch_size = 10, bits = 2, time = 1000):
    unsigned_inp = np.random.binomial(1,p,[batch_size,time//10,bits])
    #unsigned_out = 2*np.random.binomial(1,0.5,[batch_size,time//10,bits]) -1 



    #inputs = np.multiply(unsigned_inp,unsigned_out)
    inputs = unsigned_inp 
    inputs[:,0,:] = 1
    inputs = np.repeat(inputs,10,axis=1)
    #output = np.zeros_like(inputs[:,:,0])
    output = np.zeros([batch_size,time])
    for trial_idx in range(batch_size):
        
        input_1 = np.squeeze(inputs[trial_idx,:, 0])
        input_2 = np.squeeze(inputs[trial_idx,:, 1])

        for flip_idx in range(len(input_1)):
            # Get the time of the next flip
            
            '''Set the output to the sign of the flip for the
            remainder of the trial. Future flips will overwrite future
            output'''
            output[trial_idx, flip_idx] = xor(input_1[flip_idx], input_2[flip_idx])
            #output[trial_idx, flip_idx] = xor(inputs[trial_idx,flip_idx, 0], inputs[trial_idx,flip_idx, 1])

    return inputs, output

def xor_func_DELAY( p = 0.1, batch_size = 10, bits = 2, time = 1000):
    unsigned_inp = np.random.binomial(1,p,[batch_size,time//10,bits])
    #unsigned_out = 2*np.random.binomial(1,0.5,[batch_size,time//10,bits]) -1 



    #inputs = np.multiply(unsigned_inp,unsigned_out)
    inputs = unsigned_inp 
    inputs[:,0,:] = 1
    inputs = np.repeat(inputs,10,axis=1)
    #output = np.zeros_like(inputs[:,:,0])
    output_xor = np.zeros([batch_size,time])
    for trial_idx in range(batch_size):
        
        input_1 = np.squeeze(inputs[trial_idx,:, 0])
        input_2 = np.squeeze(inputs[trial_idx,:, 1])

        for flip_idx in range(len(input_1)):
            # Get the time of the next flip
            
            '''Set the output to the sign of the flip for the
            remainder of the trial. Future flips will overwrite future
            output'''
            output_xor[trial_idx, flip_idx] = xor(input_1[flip_idx], input_2[flip_idx])
            #output[trial_idx, flip_idx] = xor(inputs[trial_idx,flip_idx, 0], inputs[trial_idx,flip_idx, 1])

    output_xor[:,-20:] = 0
    output_xor[:,:20] = 0

    output = np.zeros_like(output_xor)
    for trial_idx in range(batch_size):
        input_ = np.squeeze(output_xor[trial_idx,:])
        start = np.argwhere(np.diff(input_) == 1)+1
        end = np.argwhere(np.diff(input_) == -1)+1
        for i in range(len(start)):
            #ess = min(int(end[i] + end[i]-start[i]), time)
            #output[trial_idx, int(end[i]): ess]=1
            output[trial_idx, int(end[i]): int(end[i] + end[i]-start[i])]=1
    return inputs, output



def and__delay_discrete( system_order = 5, p = 0.6, batch_size = 10, bits = 2, time = 1000):
    unsigned_inp = np.random.binomial(1,p,[batch_size,time,bits])
    #unsigned_out = 2*np.random.binomial(1,0.5,[batch_size,time//10,bits]) -1 



    #inputs = np.multiply(unsigned_inp,unsigned_out)
    inputs = unsigned_inp 
    #inputs[:,0,:] = 1
    #inputs = np.repeat(inputs,10,axis=1)
    #output = np.zeros_like(inputs[:,:,0])
    output = np.zeros([batch_size,time])
    for trial_idx in range(batch_size):
        
        input_1 = np.squeeze(inputs[trial_idx,:, 0])
        input_2 = np.squeeze(inputs[trial_idx,:, 1])

        for k in range(system_order, len(input_1) ):
            output[trial_idx, k ] = input_1[k - (system_order)]*input_2[k]
    output = output - 0.5
    return inputs, output


def and__delay( system_order = 1, p = 0.4, batch_size = 10, bits = 2, time = 100):
    unsigned_inp = np.random.binomial(1,p,[batch_size,time,bits])
    #unsigned_out = 2*np.random.binomial(1,0.5,[batch_size,time//10,bits]) -1 



    #inputs = np.multiply(unsigned_inp,unsigned_out)
    inputs = unsigned_inp 
    #inputs[:,0,:] = 1
    #inputs = np.repeat(inputs,10,axis=1)
    #output = np.zeros_like(inputs[:,:,0])
    output = np.zeros([batch_size,time])
    for trial_idx in range(batch_size):
        
        input_1 = np.squeeze(inputs[trial_idx,:, 0])
        input_2 = np.squeeze(inputs[trial_idx,:, 1])

        for k in range(system_order, len(input_1) ):
            output[trial_idx, k ] = input_1[k - (system_order)]*input_2[k]
    
    input_1_continuum = []
    input_2_continuum = []
    for trial_idx in range(batch_size):
        input_1 = np.squeeze(inputs[trial_idx,:, 0])
        input_2 = np.squeeze(inputs[trial_idx,:, 1])
        input_1_continuum_interm = []
        input_2_continuum_interm = []
        for i in range(len(input_1)):
            if input_1[i]==1:
                input_1_continuum_interm.append([1,1,1,0,0,0])
            else:
                input_1_continuum_interm.append([0,0,0,0,0,0])
        for i in range(len(input_1)):
            if input_2[i]==1:
                input_2_continuum_interm.append([1,1,1,0,0,0])
            else:
                input_2_continuum_interm.append([0,0,0,0,0,0])
        input_1_continuum.append(np.ndarray.flatten(np.array(input_1_continuum_interm)))
        input_2_continuum.append(np.ndarray.flatten(np.array(input_2_continuum_interm)))

    inputs = np.zeros((batch_size,len(input_1_continuum[0]),bits))
    for trial_idx in range(batch_size):
        inputs[trial_idx, :, 0] = np.array(input_1_continuum[trial_idx])
        inputs[trial_idx, :, 1] = np.array(input_2_continuum[trial_idx])

    output_continuum = []
    for trial_idx in range(batch_size):
        output_ = np.squeeze(output[trial_idx,:])
        output_continuum_interm = []
        for i in range(len(output_)):
            if output_[i]==1:
                output_continuum_interm.append([1,1,1,0,0,0])
            else:
                output_continuum_interm.append([0,0,0,0,0,0])
        output_continuum.append(np.ndarray.flatten(np.array(output_continuum_interm)))
########
    output = np.zeros((batch_size,len(output_continuum[0])))
    for trial_idx in range(batch_size):
        output[trial_idx, :] = np.array(output_continuum[trial_idx])
        
    return inputs, output