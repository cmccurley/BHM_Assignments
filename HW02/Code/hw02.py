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
clear = lambda: os.system('cls')
clear()

######################################################################
######################### Import Packages ############################
######################################################################

# basics
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd


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

######################################################################
############################ Questions ###############################
######################################################################
if __name__== "__main__":
    
    ############################ Question 2 ##############################
    # Shows the convergence of the binomial and poisson distributions
    # to the normal for a range of parameters
    
    # Show convergence of the Binomial distribution
    numSamples = np.arange(1,100001,500)
    distParameters = np.array([.1 ,.3 ,.5 ,.7, .9])
    
    simData = find_convergence_in_dist(numSamples, distParameters, dist='binomial')
    
    #rv = scipy.stats.binom(numTrials, numSuccess)
    #probExactly50 = scipy.stats.binom.pmf(numSuccess, numTrials, probSuccess)
    #probAtLeast50 =  scipy.stats.binom.cdf(numSuccess, numTrials, probSuccess )
    
    ############################ Question 3 ##############################
    

