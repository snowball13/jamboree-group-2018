#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 13:50:26 2018

@author: aph416
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




def import_cl5data():
    """Import the dataset from the 5th computer lab."""
    
    trainingdata = np.genfromtxt('data/cl5_training.csv', delimiter=',')
    testingdata = np.genfromtxt('data/cl5_testing.csv', delimiter=',')
    traininginputs = trainingdata[:, :3]
    trainingtargets = np.matrix(trainingdata[:, 3]).T
    testinginputs = testingdata[:, :3]
    testingtargets = np.matrix(testingdata[:, 3]).T
    
    return traininginputs, trainingtargets, testinginputs, testingtargets




def import_ethereumdata():
    """Import the ethereum dataset."""
    
    # Import the data
    trainingdata = pd.read_csv('data/ethereum_pricedata_training.csv', index_col=[0])
    testingdata = pd.read_csv('data/ethereum_pricedata_testing.csv', index_col=[0])
    
    tr_in = trainingdata[trainingdata.columns[0:6]].values
    tr_tgt = np.matrix(trainingdata['norm_future_deltaprice_24h'].values).T
    tst_in = testingdata[testingdata.columns[0:6]].values
    tst_tgt = np.matrix(testingdata['norm_future_deltaprice_24h'].values).T
    
    return tr_in, tr_tgt, tst_in, tst_tgt




def ethplot(data):
    """Make plots of the raw ethereum price data."""
    
    fig, axarray = plt.subplots(2, sharex=True)
    axarray[0].plot_date(data.index, data['high_price'], 'g-', linewidth=1, \
           markersize=1, label='Last hour\'s high')
    axarray[0].plot_date(data.index, data['low_price'], 'r-', linewidth=1, \
           markersize=1, label='Last hour\'s low')
    axarray[1].set_xlabel('Date & Time')
    axarray[0].set_ylabel('Price')
    axarray[0].legend()
    axarray[1].set_ylabel('Volume Traded')
    axarray[1].plot_date(data.index, data['volume'], '-', linewidth=1, markersize=1)
    plt.show()




def nnet_3layer(input_orig, Tran1, bias1, Tran2, bias2):
    """Calculate results from 3 layer NN: input, output, and hidden.
    
    Inputs:
    input_orig: length Ni input array. Ni = number of inputs per datapoint
    Tran1: Ni*N1 matrix containing transmission coefficients from each input to
        each node in the hidden layer. N1 = number of nodes in hidden layer
    bias1: length N1 array with biases for each node in hidden layer
    Tran2: N1*No matrix containing transmission coefficients from each node in
        hidden layer to each output. No = number of outputs
    bias2: length No array with biases for each output node
    
    Outputs:
    output_layer1: length N1 array with outputs from all nodes in hidden layer 
        (required in the backpropagation training algorithm)
    output_layer2: neural network's outputs
    """
    
    # Calculate the inputs for layer 1 (hidden layer)
    input_layer1 = np.dot(Tran1.T, input_orig) + bias1
    # Calculate the outputs
    output_layer1 = np.zeros(input_layer1.shape[0])
    for i in range(input_layer1.shape[0]):
        output_layer1[i] = 1/(1 + np.exp(-input_layer1[i]))
    
    # Calculate the inputs for layer 1 (hidden layer)
    input_layer2 = np.dot(Tran2.T, output_layer1) + bias2
    # Calculate the outputs
    output_layer2 = np.zeros(input_layer2.shape[0])
    for i in range(input_layer2.shape[0]):
        output_layer2[i] = 1/(1 + np.exp(-input_layer2[i]))
    
    return output_layer1, output_layer2
    
    


def trainiter_nnet_3layer(in_data, in_tgt, Tran1, bias1, Tran2, bias2, eta):
    """Do one training iteration of a 3 layer neural network with one output.
    
    Inputs:
    in_data: length Ni input array. Ni = number of inputs per datapoint
    in_tgt: target output value for in_data
    Tran1: Ni*N1 matrix containing transmission coefficients from each input to
        each node in the hidden layer. N1 = number of nodes in hidden layer
    bias1: length N1 array with biases for each node in hidden layer
    Tran2: N1*No matrix containing transmission coefficients from each node in
        hidden layer to each output. No = number of outputs
    bias2: length No array with biases for each output node
    eta: learning parameter: positive parameter indicating speed of update 
        along direction of most rapid descent of the error function
    
    Outputs:
    Tran1, bias1, Tran2, bias2: Updated transmission matrices and biases after
        one iteration of the gradient descent training updates
    """
    
    # Calculate the hidden layer outputs and final outputs of the neural net 
    # before any updating of transmission coefficients and biases
    output_layer1, output_layer2 = \
            nnet_3layer(in_data, Tran1, bias1, Tran2, bias2)
            
    Ni = in_data.shape[0]                      # Number of inputs per datapoint
    N1 = Tran1.shape[1]                   # Number of nodes in the hidden layer
    No = Tran2.shape[1]                       # Number of outputs per datapoint
    
    # Calculate the backpropagation coefficients. bp_cf1 is used in the 
    # update of the transition matrix from the input layer to the hidden layer,
    # and bp_cf2 is used in both the transition matrices.
    bp_cf1 = np.zeros(N1)
    for i1 in range(N1):
        bp_cf1[i1] = output_layer1[i1] * (1 - output_layer1[i1])
        
    bp_cf2 = np.zeros(No)
    for io in range(No):    
        bp_cf2[io] = (output_layer2[io] - in_tgt[io]) * output_layer2[io] * \
                (1 - output_layer2[io])
    
    # Update the transmission coefficients for Tran1
    for ii in range(Ni):
        for i1 in range(N1):
            for io in range(No):    
                Tran1[ii, i1] = Tran1[ii, i1] - \
                        eta*bp_cf2[io]*bp_cf1[i1]*Tran2[i1, io]*in_data[ii]
                       
    # Update the biases for b1
    for i1 in range(N1):
        for io in range(No):
            bias1[i1] = bias1[i1] - eta*bp_cf2[io]*bp_cf1[i1]*Tran2[i1, io]
            
    # Update the transmission coefficients for Tran2
    for i1 in range(N1):
        for io in range(No):
            Tran2[i1, io] = Tran2[i1, io] - eta*bp_cf2[io]*output_layer1[i1]
    
    # Update the biases for b2
    for io in range(No):
        bias2[io] = bias2[io] - eta*bp_cf2[io]
        
    return Tran1, bias1, Tran2, bias2
    
    

    
def trainruns_nnet_3layer(in_datas, in_tgts, N1, eta, Nt):
    """Do multiple training runs through the training dataset for a neural
    network with 3 layers (one input, one hidden, and one output layer), and 
    one output.
    
    IMPORTANT NOTE: All data entered must be numpy arrays. For example, if the
    target points are [0, 1, 0, 1], this must be entered as a column vector,
    e.g. np.array([[0], [1], [0], [1]]).
    
    Inputs:
    in_datas: Nd*Ni input array. Nd = number of datapoints, Ni = number of 
        inputs per datapoint
    in_tgts: Nd array containing target output values for in_datas
    N1: number of nodes in hidden layer
    eta: learning parameter: positive parameter indicating speed of update 
        along direction of most rapid descent of the error function
    Nt: number of full training iteratios 
    
    Outputs:
    Tran1, bias1, Tran2, bias2: Updated transmission matrices and biases after
        one iteration of the gradient descent training updates
    """
    
    Nd = in_datas.shape[0]           # Number of datapoints in training dataset
    Ni = in_datas.shape[1]                     # Number of inputs per datapoint
    
    # Use random transmissions and biases, all between -1 and 1, to start with
    Tran1 = 2*np.random.rand(Ni, N1) - np.ones((Ni, N1))
    bias1 = 2*np.random.rand(N1) - np.ones(N1)
    Tran2 = 2*np.random.rand(N1, 1) - np.ones((N1, 1))
    bias2 = 2*np.random.rand(1) - np.ones(1)
    
    # This will contain the total errors (summed over all the points in the
    # dataset) after each run through the training dataset
    total_errors = np.zeros(Nt+1)
    
    # For each run through the training dataset, errors contains the error for
    # a specific training point
    errors = np.zeros(Nd)           
    
    # For all the datapoints in the training dataset, calculate the neural net
    # outputs before any training, and input the error into the errors array 
    for id in range(Nd):
        output = nnet_3layer(in_datas[id], Tran1, bias1, Tran2, bias2)[1]
        errors[id] = (output - in_tgts[id])**2
    total_errors[0] = np.sum(errors)/in_datas.shape[0]  # Avg error for dataset
    
    # For the whole training dataset, do the training. For each datapoint in
    # the training dataset, the weights and biases are updated. There are Nt
    # runs through the dataset (which motivates the loop over range(Nt))
    for it in range(Nt):
        errors = np.zeros(Nd)                            # Array for the errors
        for id in range(Nd):
            # Calculate the updated output and use it to find the error
            output = nnet_3layer(in_datas[id], Tran1, bias1, Tran2, bias2)[1]
            errors[id] = (output - in_tgts[id])**2
            # Do the training
            Tran1, bias1, Tran2, bias2 = trainiter_nnet_3layer(in_datas[id], \
                    in_tgts[id], Tran1, bias1, Tran2, bias2, eta)
        # Average error per datapoint for a run through training dataset
        total_errors[it+1] = np.sum(errors)/in_datas.shape[0]
        
    return total_errors, Tran1, bias1, Tran2, bias2
    
    
    
    
def eta_tester(train_inputs, train_targets, N1, Nt):
    """Make plots of the error during training for various eta values.
    
    Inputs:
    train_inputs: Nd*Ni input array with the training datapoints. Nd = number 
        of datapoints, Ni = number of inputs per datapoint
    train_targets: Nd array containing target output values for train_inputs
    N1: number of nodes in hidden layer
    Nt: number of full runs through the training dataset.
    
    Outputs: None (just a plot)
    """
    
    # These are the set of training parameters that the plots are made for
    eta_vals = [0.01, 0.1, 1.0, 10.0]
    
    # Enter the errors. Each column of plots are the errors for a specific 
    # eta value, and the nth row is the error after the nth run through the
    # training dataset
    plots = np.zeros(shape=(Nt+1, len(eta_vals)))
    for k in range(len(eta_vals)):
        plots[:, k] = trainruns_nnet_3layer(train_inputs, \
                 train_targets, N1=N1, eta=eta_vals[k], Nt=Nt)[0]
    
    # Make plots of the error against Nt, for all the values of eta
    fig, ax = plt.subplots()
    for k in range(len(eta_vals)):
        labelname = 'eta = %.2f' %eta_vals[k]
        ax.plot(np.arange(Nt+1), plots[:, k], label=labelname)
    ax.set_xlabel('Number of runs through training dataset')
    ax.set_ylabel('Average error per datapoint')
    ax.set_title('%i nodes in hidden layer' %N1)
    plt.legend(ncol=4, bbox_to_anchor=(0.5, -0.1))
    plt.show()
    
    
    
    
def validate_nnettraining(tr_in, tr_tgt, tst_in, tst_tgt, N1, eta, Nt):
    """ Validate trained neural network using testing dataset by comparing
    the average error across the training and testing datasets.
    
    IMPORTANT NOTE: All data entered must be numpy arrays. For example, if the
    target points are [0, 1, 0, 1], this must be entered as a column vector,
    e.g. np.array([[0], [1], [0], [1]]).
    
    Inputs: 
    tr_in: Ndtr*Ni input array with the training datapoints. Ndtr = number of 
        training datapoints, Ni = number of inputs per datapoint
    tr_tgt: Ndtr array containing target output values for tr_in
    tst_in: Nd*Ni input array with the testing dataponts. Nd = number of 
        testing datapoints, Ni = number of inputs per datapoint
    tst_tgt: Nd array containing target output values for tst_in
    N1: number of nodes in hidden layer
    eta: training parameter
    Nt: number of full runs to do through the training dataset for training
    
    Outputs:
    tr_average_error: average error per datapoint across the training dataset
    tst_average_error: average error per datapoint across the testing dataset
    """
    
    # Do the training
    total_tr_errors, Tran1, bias1, Tran2, bias2 = \
            trainruns_nnet_3layer(tr_in, tr_tgt, N1, eta, Nt)
     
    # Errors will contain the errors for neuralnet evaluation over the testing
    # dataset
    errors_train = np.zeros(tr_in.shape[0])
    errors_test = np.zeros(tst_in.shape[0])   
            
    Nd = tst_in.shape[0]              # Number of datapoints in testing dataset

    # Will contain the outputs for the training and testing inputs respectively
    outputs_train = np.zeros(tr_in.shape[0])
    outputs_test = np.zeros(tst_in.shape[0])
    
    # Find the outputs and errors for the training dataset
    for id in range(tr_in.shape[0]):
        output = nnet_3layer(tr_in[id], Tran1, bias1, Tran2, bias2)[1]
        outputs_train[id] = output
        errors_train[id] = (output - tr_tgt[id])**2
    
    # Find the outputs and errors for the testing dataset
    for id in range(Nd):
        output = nnet_3layer(tst_in[id], Tran1, bias1, Tran2, bias2)[1]
        outputs_test[id] = output
        errors_test[id] = (output - tst_tgt[id])**2   
    
    # Output the average training and testing error
    tr_average_error = np.sum(errors_train)/tr_in.shape[0]
    tst_average_error = np.sum(errors_test)/tst_in.shape[0]
    
    return tr_average_error, tst_average_error
    
 


def validation_testing(N1):
    """Make a plot of average error per testing datapoint for a trained 
    neural network with different number of hidden layer nodes.
    
    Inputs:
    N1: maximum number of hidden layer nodes tested
    
    Outputs: None (just a graph)
    """
    
    # These will contain the training and testing errors
    tr_average_error = np.zeros(N1)
    tst_average_error = np.zeros(N1)
    
    # Input the training and testing errors
    for k in range(N1):      
        tr_average_error[k], tst_average_error[k] = \
                validate_nnettraining(g_tr_in, g_tr_tgt, g_tst_in, g_tst_tgt, \
                        N1=k, eta=1, Nt=5)
        print(k)                                    # To keep track of progress
    
    # Make a plot of training and testing errors agains the number of hidden
    # layer nodes
    fig, ax = plt.subplots()
    ax.plot(np.arange(N1) + 1, tr_average_error, label='Training dataset')
    ax.plot(np.arange(N1) + 1, tst_average_error, label='Testing dataset')
    ax.set_xlabel('Number of nodes in hidden layer')
    ax.set_ylabel('Average error per datapoint')
    ax.legend()
    plt.show()
    
    
    

def test_neuralnetwork_strategy():
    """Compare outputs given by a trained neural network, and target outputs,
    over the testing data by showing a graph"""
    
    Nd = g_tst_in.shape[0]           # Number of datatpoints in testing dataset
    
    # Find the trained neural network biases and weights
    total_tr_errors, Tran1, bias1, Tran2, bias2 = \
            trainruns_nnet_3layer(g_tst_in, g_tst_tgt, 2, 0.3, 5)
            
    # Make an array containing the outputs that the trained neural network 
    # gives when presented when all the testing outputs
    tst_outputs = np.zeros(Nd)
    for k in range(Nd):    
        tst_outputs[k] = nnet_3layer(g_tst_in[k], Tran1, bias1, Tran2, bias2)[1]

    # Plot the neural network outputs along with the target outputs for the
    # whole testing dataset
    fig, ax = plt.subplots()
    ax.plot(np.arange(Nd)+1, tst_outputs, label='trained neural net output')
    ax.plot(np.arange(Nd)+1, g_tst_tgt[:], label='target output')
    ax.set_xlabel('Datapoint from testing dataset')
    ax.set_ylabel('Output / target output')
    plt.legend()
    plt.show()