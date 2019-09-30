# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 15:52:16 2019

@author: Conma
"""

"""
***********************************************************************
    *  File:  hw02.py
    *  Name:  Connor H. McCurley
    *  Date:  2019-09-23
    *  Desc:  
**********************************************************************
"""
#Clear workspace
import os
#clear = lambda: os.system('cls')
#clear()

######################################################################
######################### Import Packages ############################
######################################################################

# basics
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math


######################################################################
######################## Funciton Definitions ########################
######################################################################

"""
***********************************************************************
    *  Func:  bin_convergence_to_normal()
    *  Desc:  Demonstrate convergence of Binomial to the Normal distribution
**********************************************************************
"""
def bin_convergence_to_normal(n, p):
    
    # generate samples from binomial
    samples = np.random.binomial(n,p,size=(1,n))
    
    # calculate theoretical mean and variance
    true_mean = p
    true_var = (1-p)*p
    
    # calculate empirical mean and variance
    est_mean = np.mean(samples)/samples.shape[1]
    est_var = np.var(samples)/samples.shape[1]
    
    ##### plot histogram overlain with expected gaussian #####
    plt.figure()
    plt.hist(samples[0,:], bins=51, density=True)
    
    # overlay expected Gaussian 
    mu = n*true_mean
    variance = n*true_var
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), c='r')
    
    plt.title(f'Binomial p={p}, n={n} \n The. Mean={true_mean}, The. Var={true_var} \n Emp. Mean={est_mean}, Emp. Var={est_var}')
    plt.show()
    
    return

"""
***********************************************************************
    *  Func:  poisson_convergence_to_normal()
    *  Desc:  Demonstrate convergence of Poisson to the Normal distribution
**********************************************************************
"""
def poisson_convergence_to_normal(n, lam):
    
    # generate samples from binomial
    samples = np.random.poisson(lam, size=(1,n))
    
    # calculate theoretical mean and variance
    true_mean = lam
    true_var = lam
    
    # calculate empirical mean and variance
    est_mean = np.mean(samples)
    est_var = np.var(samples)
    
    ##### plot histogram overlain with expected gaussian #####
    nBins = len(np.unique(samples))
    plt.figure()
    plt.hist(samples[0,:], bins=nBins, density=True)
    
    # overlay expected Gaussian 
    mu = true_mean
    variance = true_var
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), c='r')
    
    plt.title(f'Poisson lambda={lam}, \n The. Mean={true_mean}, The. Var={true_var} \n Emp. Mean={est_mean}, Emp. Var={est_var}')
    plt.show()
    
    return

"""
***********************************************************************
    *  Func:  plot_gamma()
    *  Desc:  Plot the Gamma distribution for vaious shape parameters
**********************************************************************
"""
def plot_gamma(alpha, beta):
    
    # define range to plot
    x = np.linspace(0,10,1000)
    
    # plot gamma distribution
    plt.figure()
    leg = []
    
    for alph in alpha:
        plt.plot(x, stats.gamma.pdf(x, a=alph, scale=(1/beta)))
        legEntry = 'alpha = ' + str(alph)
        leg.append(legEntry)
    plt.legend(leg)
    plt.title('Gamma Distribution')
    plt.xlim(0,6)
    
    plt.show()
    
    return


"""
***********************************************************************
    *  Func:  binomial_monte_carlo()
    *  Desc:  
**********************************************************************
"""
def binomial_monte_carlo_single_trial(p, numFlips):
    
    simData = dict()
    simData["p"] = p
    
    for n in numFlips:
        print(f'Generating data for p={p}, n={n}')
        simData[str(n)] = np.random.binomial(n, p, size = (1, 1000000)) 
    
    
    return simData

"""
***********************************************************************
    *  Func:  binomial_monte_carlo_single_trial_size()
    *  Desc:  Generate samples from a Binomial for a single MC simulation
**********************************************************************
"""
def binomial_monte_carlo(p, n):
    
    print(f'Generating data for p={p}, n={n}')
    simData = np.random.binomial(n, p, size = (100, 1000000)) 
    
    
    return simData


"""
***********************************************************************
    *  Func:  est_prop_heads()
    *  Desc:  Return the proportion of heads from Binomial samples 
**********************************************************************
"""
def est_prop_heads(simData, numFlips):
    
    for n in numFlips:
        data = simData[str(n)]
        
        # calculate proportions of heads
        propVec = data/n 
    
    return propVec


"""
***********************************************************************
    *  Func:  estimate_p_val()
    *  Desc:  Plot histogram of estimated p values from Binomial Distribution
    *         using MLE
**********************************************************************
"""
def estimate_p_val(simData, numFlips):
    
    true_p_val = simData["p"]
    
    for n in numFlips:
        data = simData[str(n)]
        
        # calculate proportions of heads
        propVec = data/n 
        
        # estimate p value
        mle_p_est = np.sum(propVec)/propVec.shape[1] 
        
        plt.figure()
        plt.hist(data[0,:], bins=n, density=True)
        plt.title(f'Histogram of Binomial Trials \n Number of Flips={n} \n True p Value={true_p_val} \n MLE Estimate={mle_p_est:.4f}')
        
        
    return


"""
***********************************************************************
    *  Func:  compute_point_est_std_error()
    *  Desc:  Find the point estimate standard deviation from a single
    *         MC trial
**********************************************************************
"""
def compute_point_est_std_error(simData, numFlips):
    
    point_error = dict()
    
    # generate 100 MC simulations with 1000000 samples each
    for n in numFlips: 
        p = simData["p"]
        data = simData[str(n)]
        data = data/n
#        mean_est = np.mean(data)
        N = data.shape[1]
        s_y_squared = (1/(N-1))*np.sum((data-p)**2)
        
        point_error[str(n)] = np.sqrt(s_y_squared)
    
    return point_error

"""
***********************************************************************
    *  Func:  compute_mc_std_error()
    *  Desc:  Find the mean and variance of parameter estimate error
    *         for each of the MC simulations
**********************************************************************
"""
def compute_mc_std_error(p, numFlips):
    
    estData = dict()
    
    # generate 100 MC simulations with 1000000 samples each
    for n in numFlips:    
        data = binomial_monte_carlo(p, n)
        data = data/n
        
        # calculate statistics of estimate error
        mean_est = np.mean(data, axis=1) - p
        mean_of_mean_est = np.mean(mean_est-p)
        std_of_mean_est = np.std(mean_est-p)
        
        # save data
        n_summary = dict()
        n_summary["mean_est"] = mean_est
        n_summary["mean_of_mean_est"] = mean_of_mean_est
        n_summary["std_of_mean_est"] = std_of_mean_est
    
        estData[str(n)] = n_summary
        
    return estData

"""
***********************************************************************
    *  Func:  plot_error_hist()
    *  Desc:  Plot histogram of MC errors 
**********************************************************************
"""
def plot_error_hist(p, data):
    
    # plot histogram of error for the mean estimates
    for n in data.keys():    
        
        mean_est = data[str(n)]["mean_est"]
        mean_of_mean_est = np.mean(mean_est)
        std_of_mean_est = data[str(n)]["std_of_mean_est"]
        
        # plot histogram
        plt.figure()
        plt.hist(mean_est, bins='auto', density=True)
        plt.title(f'Histogram of Estimate Errors for 100 MC Simulations \n Number of Flips={n} \n Mean={mean_of_mean_est:.2f}, STD={std_of_mean_est:.5f}')
        
    return

"""
***********************************************************************
    *  Func:  plot_Q6_trajectories()
    *  Desc:  Plot dN/dt for various combinations of r and K
**********************************************************************
"""
def plot_Q6_trajectories(rList, KList):
    
    plt.figure()
    N = np.linspace(0,15,100)
    leg = []
    
    for r in rList:
        for K in KList:
            
            # evaluate differential equation
            y = (r*N)*(1-(N/K))
        
            # plot trajectory
            plt.plot(N, y)
            legEntry = 'r= ' + str(r) + ', K= ' + str(K)
            leg.append(legEntry)
    plt.legend(leg)
    plt.title('Trajectories for dN/dt', fontsize=16)
    plt.xlabel('N', fontsize=14)
    plt.ylabel('dN/dt', fontsize=14)
    plt.ylim(-5,30)
    
    return

"""
***********************************************************************
    *  Func:  Q6_observation_error()
    *  Desc:  Generate plot of 10**3 sample paths with observation error
**********************************************************************
"""
def Q6_observation_error(r, K):
    
    numSamples = 200  # length of sample path
    numTrials = 10000 # number of sample paths
    n = np.zeros((numTrials, numSamples))
    n[:,0] = 0.5
    tRange = np.arange(0,numSamples)
    
    
    plt.figure()
    
    for trial in range(0, numTrials):
        # print status
        if not(trial % 1000):
            print(f'Trial {trial} of {numTrials}')
            
        # generate sample path N[t] = N[t-1] + r*N[t-1]*(1-(N[t-1]/K)) + M
        for t in range(1, numSamples):
            # generate process sample
            n[trial, t] = (n[trial,t-1] + (r*n[trial, t-1])*(1-(n[trial,t-1]/K)))
            
        # add observation error
        n[trial, :] = n[trial, :] + np.random.normal(0,0.5, size=(1,numSamples))
            
        plt.plot(tRange, n[trial, :], color='black')
        
    # plot sample paths
    plt.xlim(0,numSamples)
    plt.ylim(-5,10)
    
    plt.title(f'{numTrials} Sample Paths with Observation Error')
    
    return

"""
***********************************************************************
    *  Func:  Q6_process_error()
    *  Desc:  Generate plot of 10**3 sample paths with process error
**********************************************************************
"""
def Q6_process_error(r, K):
    
    numSamples = 200  # length of sample path
    numTrials = 10000 # number of sample paths
    n = np.zeros((numTrials, numSamples))
    n[:,0] = 0.5
    tRange = np.arange(0,numSamples)
    
    
    plt.figure()
    
    for trial in range(0, numTrials):
        # print status
        if not(trial % 1000):
            print(f'Trial {trial} of {numTrials}')
            
        # generate sample path N[t] = N[t-1] + r*N[t-1]*(1-(N[t-1]/K)) + M
        for t in range(1, numSamples):
            
            # generate sample with process error
            n[trial, t] = (n[trial,t-1] + (r*n[trial, t-1])*(1-(n[trial,t-1]/K))) + np.random.normal(0,0.5)
        
        plt.plot(tRange, n[trial, :], color='black')
        
    # plot sample paths
    plt.xlim(0,numSamples)
    plt.ylim(-5,10)
    
    plt.title(f'{numTrials} Sample Paths with Process Error')
    
    return

######################################################################
############################ Questions ###############################
######################################################################
if __name__== "__main__":
    
    ############################ Question 2 ##############################
    # Shows the convergence of the binomial and poisson distributions
    # to the normal for a range of parameters
    
    ##### Show convergence of the Binomial to Normal distribution #####
    numSamples = 1000000
    p = 0.5
#    bin_convergence_to_normal(numSamples, p)
    
    
    ##### Show convergence of the Poisson distribution to Normal #####
    numSamples = 1000000
    lam = 100
#    poisson_convergence_to_normal(numSamples, lam)
    
    ############################ Question 3 ##############################
    # plot the gamma distribution
    
    # define parameters
    beta = 4
    alpha = [0.1, 0.5, 1, 5, 10]
    
    # plot distribution
#    plot_gamma(alpha, beta)
    
    
    ############################ Question 4 ##############################
    # run Monte Carlo simulations for flipping a coin with a known bias
    
    # set parameters
    p = 0.3
    numFlips = [5, 10, 50, 100]
    
    # generate data
#    simData = binomial_monte_carlo_single_trial(p, numFlips)
#    np.save('Q4_data.npy', simData, allow_pickle=True)
    
    # load data
    simData = np.load('Q4_data.npy', allow_pickle=True).item()
    
    # estimate bounds for expected value of each trial
#    est_prop_heads(simData, numFlips)
    
    # compute standard errors 
#    point_error = compute_point_est_std_error(simData, numFlips)
        
    
    
    ############################ Question 5 ##############################
    # compute MLE estimate for p parameter of binomial
    
    # load data
    simData = np.load('Q4_data.npy', allow_pickle=True).item()
    
    
    # compute the MLE estimates
#    estimate_p_val(simData, numFlips)
    
    # compute Monte Carlo standard errors
#    error_data = compute_mc_std_error(p, numFlips)
    
    # plot error histograms
#    plot_error_hist(p, error_data)
#    
    ############################ Question 6 ##############################
    
    # plot simulated trajectories for r and K
    r = [0.1, 1, 5, 10]
    K = [0.1, 1, 5, 10]
    
#    plot_Q6_trajectories(r, K)
    
    
    # descritize the model
    r = 0.1
    K = 1
    
    # plot trajectories with observation error
#    Q6_observation_error(r, K)
    
    # plot trajectories with process error
#    Q6_process_error(r, K)
    
    
    
    
    
    

