# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 15:52:16 2019

@author: Conma
"""

"""
***********************************************************************
    *  File:  hw02.py
    *  Name:  Connor H. McCurley
    *  Date:  2019-08-30
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
import pandas as pd
import math


######################################################################
######################## Funciton Definitions ########################
######################################################################
def find_convergence_in_dist(numSamples, distParameters, dist):
    
    simData = dict()
    
    sim = 1
    numSims = len(distParameters)  
    
    for p in distParameters:
        dataStruct = dict()
        
        # update status of simulations
        print(f'Simulation {sim} of {numSims}')
        p_mean_est = []
        p_var_est = []
        pVal = p
        
        # get mean estimate and variance of mean estimate for a range of sample values, 100 trials each.
        for n in numSamples:
            if(dist=='binomial'):
                samples = np.random.binomial(n,p,size=(100,n))
                mean_est = np.mean(samples, axis=1)
                var_of_mean_est = np.var(mean_est)
                mean_of_mean_est = np.mean(mean_est)
                
#            elif(dist=='poisson'):
                p_mean_est.append(mean_of_mean_est)
                p_var_est.append(var_of_mean_est)
        
        # add simulation over n to structure of all trials
        key = 'sim' + str(sim)
        dataStruct["p_mean_est"] = p_mean_est
        dataStruct["p_var_est"] = p_var_est
        dataStruct["pVal"] = pVal
        simData[key] = dataStruct
        sim = sim + 1
        
    return dataStruct

"""
***********************************************************************
    *  Func:  bin_convergence_to_normal()
    *  Desc:  
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
    *  Desc:  
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
    *  Desc:  
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
    
    plt.show()
    
    return


"""
***********************************************************************
    *  Func:  binomial_monte_carlo()
    *  Desc:  
**********************************************************************
"""
def binomial_monte_carlo(p, numFlips):
    
    simData = dict()
    simData["p"] = p
    
    for n in numFlips:
        print(f'Generating data for p={p}, n={n}')
        simData[str(n)] = np.random.binomial(n, p, size = (1, 1000000)) 
    
    
    return simData


"""
***********************************************************************
    *  Func:  est_prop_heads()
    *  Desc:  
**********************************************************************
"""
def est_prop_heads(simData, numFlips):
    
    for n in numFlips:
        data = simData[str(n)]
        
        # calculate proportions of heads
        propVec = data/n 
        
        # get mean of proportions
        meanPropVec = np.mean(propVec)
        
        
        # get deviation within 95%
    
    return 


"""
***********************************************************************
    *  Func:  estimate_p_val()
    *  Desc:  
**********************************************************************
"""
def estimate_p_val(simData, numFlips):
    mle_est = dict()
    
    true_p_val = simData["p"]
    
    for n in numFlips:
        data = simData[str(n)]
        
        # calculate proportions of heads
        propVec = data/n 
        
        # estimate p value
        mle_p_est = np.sum(propVec)/propVec.shape[1] 
        
        plt.figure()
        plt.hist(data[0,:], bins=n, density=True)
        plt.title(f'Histogram of Binomial Trials \n Number of Flips={n} \n True p Value={true_p_val} \n MLE Estimate={mle_p_est}')
        
        
    return mle_p_est

######################################################################
############################ Questions ###############################
######################################################################
if __name__== "__main__":
    
    ############################ Question 2 ##############################
    # Shows the convergence of the binomial and poisson distributions
    # to the normal for a range of parameters
    
    ##### Show convergence of the Binomial to Normal distribution #####
    numSamples = 100000
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
#    simData = binomial_monte_carlo(p, numFlips)
#    np.save('Q4_data.npy', simData, allow_pickle=True)
    
    # load data
    simData = np.load('Q4_data.npy', allow_pickle=True).item()
    
    # estimate bounds for expected value of each trial
#    est_prop_heads(simData, numFlips)
    
    
    ############################ Question 5 ##############################
    # compute MLE estimate for p parameter of binomial
    
    # load data
    simData = np.load('Q4_data.npy', allow_pickle=True).item()
    
    
    # compute the MLE estimates
    mle_est = estimate_p_val(simData, numFlips)
    
    ############################ Question 6 ##############################
    
    
    
    
    
    
    
    

