# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 15:52:16 2019

@author: Conma
"""

"""
***********************************************************************
    *  File:  hw01.py
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


############################ Question 2 ##############################

#numTrials = 100
#probSuccess = 0.5
#numSuccess = 50
#
#rv = scipy.stats.binom(numTrials, numSuccess)
#probExactly50 = scipy.stats.binom.pmf(numSuccess, numTrials, probSuccess)
#probAtLeast50 =  scipy.stats.binom.cdf(numSuccess, numTrials, probSuccess )

############################ Question 2 ##############################

points = 4
mean = 3.5
var =  14^2

rv = scipy.stats.norm(mean, var)
prob4 = rv.cdf(points)
probWin = 1 - prob4 
