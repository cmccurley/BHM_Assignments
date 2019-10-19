### Stan demo

# clear environment
rm(list=ls())

library("rstan")
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)
Sys.setenv(LOCAL_CPPFLAGS = '-march=native')


##########################################################
##################### Question 1 Problem 1 ###############
##########################################################
# 
# tauVect = c(0.1, 0.5, 1, 5, 10, 50, 100)
# 
# # initialize theta matrix
# thetaMat = matrix(, nrow = 7, ncol = 8)
# 
# idx = 1
# 
# for (t in tauVect){
# 
# # load data
#   schools_dat <- list(J = 8,
#                       y = c(28,  8, -3,  7, -1,  1, 18, 12),
#                       sigma = c(15, 10, 16, 11,  9, 11, 10, 18),
#                       tau = t)
# 
#   # fit stan model
#   fit <- stan(file = '8schools_q1_p1.stan', data = schools_dat)
#   print(fit)
# 
#   # access mean estamtes for each school mean (theta)
#   theta_summary =  summary(fit, pars = c("theta"))$summary
#   thetas = theta_summary[,1]
#   thetaMat[idx,] = thetas
# 
#   idx = idx+1
# }
# 
# ##### plot theta estimates for each school for varying taus #####
# # generate random colors for plotting
# cl <- rainbow(8)
# colVect = matrix(, nrow = 8, ncol = 0)
# colVect[1] = cl[1]
# 
# # create plot
# plot.new
# plot(tauVect,thetaMat[,1], type='l',col=cl[1], lwd=2, xlab="Tau",ylab="Theta Value", xlim=c(0,100), ylim=c(-10, 40))
# # axis(1, at=tauVect)
# 
# for (idx in 2:length(thetaMat[1,])){
#   lines(tauVect,thetaMat[,idx], col=cl[idx],lwd=2)
#   colVect[idx] = cl[idx]
# }
# title(main="Theta Est for Fixed Tau")
# legend(x=0,y=40, schools_dat$y ,col = colVect, lwd = 1, cex = 0.8)

##########################################################
##################### Question 1 Problem 2 ###############
##########################################################
# # define scale vector
# scaleVect = c(1,5,10,50)
# 
# # initialize theta matrix
# thetaMat = matrix(, nrow = length(scaleVect), ncol = 8)
# idx = 1
# 
# for (s in scaleVect){
# 
#   # load data
#   schools_dat <- list(J = 8,
#                       y = c(28,  8, -3,  7, -1,  1, 18, 12),
#                       sigma = c(15, 10, 16, 11,  9, 11, 10, 18),
#                       scale = s)
# 
#   # fit stan model
#   fit <- stan(file = '8schools_q1_p2.stan', data = schools_dat)
#   print(fit)
# 
#   # access mean estamtes for each school mean (theta)
#   theta_summary =  summary(fit, pars = c("theta"))$summary
#   thetas = theta_summary[,1]
#   thetaMat[idx,] = thetas
# 
#   idx = idx+1
# }
# 
# ##### plot theta estimates for each school for varying taus #####
# # generate random colors for plotting
# cl <- rainbow(8)
# colVect = matrix(, nrow = 8, ncol = 0)
# colVect[1] = cl[1]
# 
# # create plot
# plot.new
# plot(scaleVect,thetaMat[,1], type='l',col=cl[1], lwd=2, xlab="Scale of Variance on Tau Prior",ylab="Theta Value", xlim=c(0,50), ylim=c(0, 15))
# # axis(1, at=tauVect)
# 
# for (idx in 2:length(thetaMat[1,])){
#   lines(scaleVect,thetaMat[,idx], col=cl[idx],lwd=2)
#   colVect[idx] = cl[idx]
# }
# title(main="Theta Est for Est Tau")
# legend(x=0,y=15,schools_dat$y,col = colVect, lwd = 1, cex = 0.5)


##########################################################
##################### Question 2 Problem 1 ###############
##########################################################
# 
# ### Simulating fake logistic growth data
# 
# # Define Parameters
# proc_error = 5
# N = 50
# init = 20
# K = 100
# rVect = c(0.1, 0.5, 1)
# 
# # Simulate process with no error for 3 different r values
# y = matrix(NA, nrow=3, ncol=N)
# 
# y[,1] = init;
# 
# for (idx in 1:length(rVect)){
#   r = rVect[idx]
#   for(i in 2:N){
#     y[idx,i] <- y[idx,i-1] + r*y[idx,i-1]*(1-y[idx,i-1]/K)
#   }
# }
# 
# obs_N = seq(1,N,1)
# 
# # plot logistic curve with process error
# plot.new
# plot(obs_N, y[1,], type='l',col='blue', lwd=2, xlab="N",ylab="Process Value", xlim=c(0,N))
# lines(obs_N, y[2,],col='green', lwd=2)
# lines(obs_N, y[3,],col='red', lwd=2)
# title(main="Logistic Growth")
# legend("bottomright", "(x,y)",rVect,col = c("blue", "green", "red"), lwd = 1, cex = 1, title="R")

##########################################################
##################### Question 2 Problem 2 ###############
##########################################################
# 
# ### Simulating fake logistic growth data with observation error
# 
# # Define Parameters
# proc_error = 5
# N = 50
# init = 20
# K = 100
# rVect = c(0.1, 0.5, 1)
# obs_error = 2
# 
# # Simulate process with observation error for 3 different r values
# y = matrix(NA, nrow=3, ncol=N)
# 
# y[,1] = init;
# 
# for (idx in 1:length(rVect)){
#   r = rVect[idx]
#   for(i in 2:N){
#     y[idx,i] <- y[idx,i-1] + r*y[idx,i-1]*(1-y[idx,i-1]/K)
#     y[idx,i] = y[idx,i] + rnorm(1,0,obs_error)
#   }
# }
# 
# obs_N = seq(1,N,1)
# 
# # plot logistic curve with process error
# plot.new
# plot(obs_N, y[1,], type='p',col='blue', xlab="N",ylab="Process Value", xlim=c(0,N))
# points(obs_N, y[2,],col='green')
# points(obs_N, y[3,],col='red')
# title(main="Logistic Growth with Observation Error Variance 2")
# legend("bottomright", "(x,y)",rVect,col = c("blue", "green", "red"), lwd = 1, cex = 1, title="R")

##########################################################
####################### Question 3 #######################
##########################################################
# 
# # Run stan simulations for the three r values 
# # and plot histograms 
# 
# # Define Parameters
# N = 50
# init = 20
# K = 100
# rVect = c(0.1, 0.5, 1)
# obs_error = 2
# 
# # Simulate process with observation error for 3 different r values
# # Generate data
# y = matrix(NA, nrow=3, ncol=N)
# 
# y[,1] = init;
# 
# for (idx in 1:length(rVect)){
#   r = rVect[idx]
#   for(i in 2:N){
#     y[idx,i] <- y[idx,i-1] + r*y[idx,i-1]*(1-y[idx,i-1]/K)
#     y[idx,i] = y[idx,i] + rnorm(1,0,obs_error)
#   }
# }
# 
# obs_N = seq(1,N,1)
# 
# ###### Simulation for r = 0.1 #############
# 
# # define data
# stan_data <- list(N=N,y=y[1,])
# 
# # initial conditions
# initf1 <- function() {
#   list(r = 0.1, K = 100, mu = array(1,dim=c(1,100)), obs_error = 2)
# } 
# 
# # instantiate model
# logistic_ss1 <- stan(file = "logistic_Bayes_q2_p_3_obs_error_only.stan", data = stan_data,
#                      chains = 6, iter = 1000, init = initf1)
# 
# mu_infer <- extract(logistic_ss1, pars = c("mu"))[[1]]
# 
# post_draws <- as.data.frame(t(mu_infer))
# library(reshape2)
# post_melt <- melt(post_draws)
# 
# # Plotting model fit to data with posterior draws 
# # of the latent variable 
# ggplot() + geom_point(data=data.frame(x=obs_N,y=y[1,]),aes(x=x,y=y),color="red") + 
#     ggtitle("Data Realizations for r=0.1")
# 
# # extract estimated parameters for r, K, and observation error variance
# par_infer <- extract(logistic_ss1, pars = c("r", "K", "obs_error"))
# 
# # Plot histogram for r
# r_post <- ggplot(data = data.frame(x = par_infer[[1]]),aes(x)) + stat_density(alpha = 0.5) +
#   geom_vline(aes(xintercept = 0.1),color = "red") + ggtitle("Histogram of r for a true value of 0.1")
# 
# print(r_post)
# 
# # plot histogram for observation error variance 
# obsError_post <- ggplot(data = data.frame(x = par_infer[[3]]),aes(x)) + stat_density(alpha = 0.5) +
#   geom_vline(aes(xintercept = obs_error),color = "red") + ggtitle("Histogram of Obs. Error Variance for a true value of 2")
# 
# print(obsError_post)
# 
# ###### Simulation for r = 0.5 #############
# 
# # define data
# stan_data <- list(N=N,y=y[2,])
# 
# # initial conditions
# initf1 <- function() {
#   list(r = 0.5, K = 100, mu = array(1,dim=c(1,100)), obs_error = 2)
# } 
# 
# # instantiate model
# logistic_ss1 <- stan(file = "logistic_Bayes_q2_p_3_obs_error_only.stan", data = stan_data,
#                      chains = 6, iter = 1000, init = initf1)
# 
# mu_infer <- extract(logistic_ss1, pars = c("mu"))[[1]]
# 
# post_draws <- as.data.frame(t(mu_infer))
# library(reshape2)
# post_melt <- melt(post_draws)
# 
# # Plotting model fit to data with posterior draws 
# # of the latent variable 
# ggplot() + geom_point(data=data.frame(x=obs_N,y=y[2,]),aes(x=x,y=y),color="red") + 
#   ggtitle("Data Realizations for r=0.5")
# 
# # extract estimated parameters for r, K, and observation error variance
# par_infer <- extract(logistic_ss1, pars = c("r", "K", "obs_error"))
# 
# # Plot histogram for r
# r_post <- ggplot(data = data.frame(x = par_infer[[1]]),aes(x)) + stat_density(alpha = 0.5) +
#   geom_vline(aes(xintercept = 0.5),color = "red") + ggtitle("Histogram of r for a true value of 0.5")
# 
# print(r_post)
# 
# # plot histogram for observation error variance 
# obsError_post <- ggplot(data = data.frame(x = par_infer[[3]]),aes(x)) + stat_density(alpha = 0.5) +
#   geom_vline(aes(xintercept = obs_error),color = "red") + ggtitle("Histogram of Obs. Error Variance for a true value of 2")
# 
# print(obsError_post)
# 
# ###### Simulation for r = 1 #############
# 
# # define data
# stan_data <- list(N=N,y=y[3,])
# 
# # initial conditions
# initf1 <- function() {
#   list(r = 1, K = 100, mu = array(1,dim=c(1,100)), obs_error = 2)
# } 
# 
# # instantiate model
# logistic_ss1 <- stan(file = "logistic_Bayes_q2_p_3_obs_error_only.stan", data = stan_data,
#                      chains = 6, iter = 1000, init = initf1)
# 
# mu_infer <- extract(logistic_ss1, pars = c("mu"))[[1]]
# 
# post_draws <- as.data.frame(t(mu_infer))
# library(reshape2)
# post_melt <- melt(post_draws)
# 
# # Plotting model fit to data with posterior draws 
# # of the latent variable 
# ggplot() + geom_point(data=data.frame(x=obs_N,y=y[3,]),aes(x=x,y=y),color="red") + 
#   ggtitle("Data Realizations for r=1") 
# 
# # extract estimated parameters for r, K, and observation error variance
# par_infer <- extract(logistic_ss1, pars = c("r", "K", "obs_error"))
# 
# # Plot histogram for r
# r_post <- ggplot(data = data.frame(x = par_infer[[1]]),aes(x)) + stat_density(alpha = 0.5) +
#   geom_vline(aes(xintercept = 1),color = "red") + ggtitle("Histogram of r for a true value of 1")
# 
# print(r_post)
# 
# # plot histogram for observation error variance 
# obsError_post <- ggplot(data = data.frame(x = par_infer[[3]]),aes(x)) + stat_density(alpha = 0.5) +
#   geom_vline(aes(xintercept = obs_error),color = "red") + ggtitle("Histogram of Obs. Error Variance for a true value of 2")
# 
# print(obsError_post)

##########################################################
##################### Question 4 Problem 1 ###############
##########################################################
# simulate logistic growth with gamma process variance and 
# gaussian observation variance


# clear environment
rm(list=ls())

library("rstan")
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)
Sys.setenv(LOCAL_CPPFLAGS = '-march=native')

# Define Parameters
N = 100
r = 0.5
n_sample_paths = 6
init = 5
K = 100
dk = 0.25
obs_error = 2
obs_error_fixed = 0.8
obs_N = seq(1,N,1)


for(j in 1:M) {
  for(i in 2:N){
    size3[j,i] <- rgamma(1,shape=(size3[j,i-1] + r*size3[j,i-1]*(1-(size3[j,i-1]/K)))/dk,rate=1/dk);
  }
}


# Simulate process with observation error for 3 different r values
# Generate data

######### process error only #############
y = matrix(NA, nrow=n_sample_paths, ncol=N);

y[,1] = init;

for (idx in 1:n_sample_paths){
  for(i in 2:N){
    y[idx,i] <- rgamma(1,shape=(y[idx,i-1] + r*y[idx,i-1]*(1-(y[idx,i-1]/K)))/dk,rate=1/dk);
  }
}

y_proc_error = y;



# Plot sample paths
color = c("red", "green", "blue", "#999999", "#E69F00", "#56B4E9")

ggplot() + geom_point(data=data.frame(x=obs_N,y=y_proc_error[1,]),aes(x=x,y=y),color=color[1]) +
  geom_point(data=data.frame(x=obs_N,y=y_proc_error[2,]),aes(x=x,y=y),color=color[2]) +
  geom_point(data=data.frame(x=obs_N,y=y_proc_error[3,]),aes(x=x,y=y),color=color[3]) +
  geom_point(data=data.frame(x=obs_N,y=y_proc_error[4,]),aes(x=x,y=y),color=color[4]) +
  geom_point(data=data.frame(x=obs_N,y=y_proc_error[5,]),aes(x=x,y=y),color=color[5]) +
  geom_point(data=data.frame(x=obs_N,y=y_proc_error[6,]),aes(x=x,y=y),color=color[6]) +
  theme_bw() + ggtitle('Sample Paths with Process Error only \n K=100, r=0.5, dK=0.25') +
  xlab("N") + ylab("Process Value")



###### process and observation error ##########
y = y_proc_error;

for (idx in 1:n_sample_paths){
  for(i in 2:N){
    y[idx,i] <- rnorm(1,y[idx,i],obs_error);
  }
}
  
y_proc_and_obs = y;

color = c("red", "green", "blue", "#999999", "#E69F00", "#56B4E9")

ggplot() + geom_point(data=data.frame(x=obs_N,y=y_proc_and_obs[1,]),aes(x=x,y=y),color=color[1]) +
  geom_point(data=data.frame(x=obs_N,y=y_proc_and_obs[2,]),aes(x=x,y=y),color=color[2]) +
  geom_point(data=data.frame(x=obs_N,y=y_proc_and_obs[3,]),aes(x=x,y=y),color=color[3]) +
  geom_point(data=data.frame(x=obs_N,y=y_proc_and_obs[4,]),aes(x=x,y=y),color=color[4]) +
  geom_point(data=data.frame(x=obs_N,y=y_proc_and_obs[5,]),aes(x=x,y=y),color=color[5]) +
  geom_point(data=data.frame(x=obs_N,y=y_proc_and_obs[6,]),aes(x=x,y=y),color=color[6]) +
  theme_bw()  + ggtitle('Sample Paths with Process Error and Observation Error \n K=100, r=0.5, dK=0.25, obs_error=2')  +
  xlab("N") + ylab("Process Value")


###### process and fixed observation error ##########
y = y_proc_error;

for (idx in 1:n_sample_paths){
  for(i in 2:N){
    y[idx,i] <- y[idx,i] + obs_error_fixed
  }
}
  
y_proc_and_fixed_obs =  y

# Plot Sample Paths
color = c("red", "green", "blue", "#999999", "#E69F00", "#56B4E9")

ggplot() + geom_point(data=data.frame(x=obs_N,y=y_proc_and_fixed_obs[1,]),aes(x=x,y=y),color=color[1]) +
  geom_point(data=data.frame(x=obs_N,y=y_proc_and_fixed_obs[2,]),aes(x=x,y=y),color=color[2]) +
  geom_point(data=data.frame(x=obs_N,y=y_proc_and_fixed_obs[3,]),aes(x=x,y=y),color=color[3]) +
  geom_point(data=data.frame(x=obs_N,y=y_proc_and_fixed_obs[4,]),aes(x=x,y=y),color=color[4]) +
  geom_point(data=data.frame(x=obs_N,y=y_proc_and_fixed_obs[5,]),aes(x=x,y=y),color=color[5]) +
  geom_point(data=data.frame(x=obs_N,y=y_proc_and_fixed_obs[6,]),aes(x=x,y=y),color=color[6]) +
  theme_bw()  + ggtitle('Sample Paths with Process Error and Fixed Observation Error \n K=100, r=0.5, dK=0.25, fixed_obs_error=0.8')  +
  xlab("N") + ylab("Process Value")


########################## Run MCMC Simulations ####################################
# clear environment
rm(list=ls())

library("rstan")
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)
Sys.setenv(LOCAL_CPPFLAGS = '-march=native')

################### Process error only #####################################
y = y_proc_error
stan_data <- list(N=N,KK=n_sample_paths,M=n_sample_paths,y=y,times=obs_N)

fit <- stan(file = "logistic5_Bayes.stan", data = stan_data, chains = 6, iter = 2000,
                      seed = 12345)

print(fit) # promising 

traceplot(fit, pars = c("obs_error", "K", "r", "k"))


################### Process and Observation Error ##########################



################### Process and Fixed Observation Error #####################



