### Simulating fake logistic growth data 

# Define Parameters
proc_error = 5
N = 50
init = 20
r = 0.1 
K = 100

# Simulate process with process error
y = rep(NA,N); 
y[1] = init;
for(i in 2:N){
  y[i] <- y[i-1] + r*y[i-1]*(1-y[i-1]/K) + rnorm(1,0,proc_error)
}

# plot logistic curve with process error
obs_N = seq(1,50,1)
plot(obs_N,y)



# N <- 100 # Length of simulation 
# r <- 0.1 # intrinsic growth rate
# K <- 100 # carrying capacity 
# init <- 1 # initial condition 
# 
# size <- rep(NA,N); 
# size[1] <- init; 
# for(i in 2:N){
#   size[i] <- size[i-1] + r*size[i-1]*(1-size[i-1]/K)
# }
# 
# # Now simulating observation error
# obs_error <- 5
# proc_error <- 5
# size_obs <- rep(NA,length(size));
# size_obs <- size + rnorm(N,0,obs_error)
# 
# plot(size_obs) # Note the problem with negative values! 
# 
# # Thinning observations down to 20 
# obs_N <- sort(sample(seq(1,100,1),15))
# obs_sizes <- size_obs[obs_N]
# plot(obs_N,obs_sizes)
# 
# # Putting in list for Stan
# stan_data <- list(N=N,M=4,y=obs_sizes,times=obs_N)
# 
# 
# # Fitting model 
# 
# library(rstan)
# options(mc.cores = parallel::detectCores())
# rstan_options(auto_write = TRUE)
# Sys.setenv(LOCAL_CPPFLAGS = '-march=native')
# 
# initf1 <- function() {
#   list(r = 0.2, K = 50, mu = array(1,dim=c(1,100)), obs_error = 1, proc_error = 2)
# } 
# 
# 
# logistic_ss1 <- stan(file = "logistic_Bayes_process_error.stan", data = stan_data,
#                      chains = 6, iter = 1000, init = initf1)
# 
# 
# #print(logistic_ss1)
# mu_infer <- extract(logistic_ss1, pars = c("mu"))[[1]]
# 
# post_draws <- as.data.frame(t(mu_infer))
# library(reshape2)
# post_melt <- melt(post_draws)
# # Identifying the times 
# post_melt$time <- rep(seq(1,100,1),3000)
# 
# # Plotting model fit to data with posterior draws 
# # of the latent variable 
# ggplot() + geom_point(data=data.frame(x=obs_N,y=obs_sizes),aes(x=x,y=y),color="red") +
#   geom_line(data=post_melt,aes(x=time,y=value,group=variable),size = 0.004) +
#   theme_bw()
# 
# # Plotting inference on the r, K, sigma_obs parameters 
# 
# par_infer <- extract(logistic_ss1, pars = c("r", "K", "proc_error"))
# 
# # r 
# r_post <- ggplot(data = data.frame(x = par_infer[[1]]),aes(x)) + stat_density(alpha = 0.5) +
#   geom_vline(aes(xintercept = r),color = "red")
# print(r_post)
# 
# # K 
# K_post <- ggplot(data = data.frame(x = par_infer[[2]]),aes(x)) + stat_density(alpha = 0.5) +
#   geom_vline(aes(xintercept = K),color = "red")
# print(K_post)
# 
# # obs error 
# obsError_post <- ggplot(data = data.frame(x = par_infer[[3]]),aes(x)) + stat_density(alpha = 0.5) +
#   geom_vline(aes(xintercept = obs_error),color = "red")
# print(obsError_post)
# 
# # proc error
# procError_post <- ggplot(data = data.frame(x = par_infer[[4]]),aes(x)) + stat_density(alpha = 0.5) +
#   geom_vline(aes(xintercept = obs_error),color = "red")
# print(procError_post)
# 
# 
# 
# ## Model with process AND observation error 
# 
# size2 <- rep(NA,N); 
# size2[1] <- init; 
# for(i in 2:N){
#   size2[i] <- size2[i-1] + r*size2[i-1]*(1-size2[i-1]/K) + rnorm(1,0,0.5)
# }
# 
# plot(size2)
# 
# # Now simulating observation error
# obs_error <- 5
# size_obs2 <- matrix(NA,3,length(size2));
# for(i in 1:N){
# size_obs2[,i] <- size2[i] + rnorm(3,0,obs_error)
# }
# 
# # Thinning observations down to 30 
# obs_N2 <- sort(sample(seq(1,100,1),30))
# obs_sizes2 <- size_obs2[,obs_N2]
# 
# plot(obs_N2, obs_sizes2[1,])
# points(obs_N2, obs_sizes2[2,],col=2,add=T)
# points(obs_N2, obs_sizes2[3,],col=3,add=T)
# 
# 
# # Putting in list for Stan
# stan_data2 <- list(N=N,KK=3,M=length(obs_N2),y=obs_sizes2,times=obs_N2-1)
# 
# initf2 <- function() {
#   list(r = 0.2, K = 50, mu = array(1,dim=c(1,100)), obs_error = 1,
#        proc_error = 1)
# } 
# 
# logistic_ss2 <- stan(file = "logistic2_Bayes.stan", data = stan_data2,
#                      chains = 1, iter = 1000, init = initf2)
# 
# 
# 
# print(logistic_ss2)
# 
# logistic_ss2b <- stan(fit = logistic_ss2, data = stan_data2, chains = 6, iter = 2000,
#                       seed = 12345)
# 
# print(logistic_ss2b)
# 
# traceplot(logistic_ss2b, pars = c("r","K","proc_error")) 
# # Notice the very poor mixing on the process error chains (and this is with multiple observations!
# 
# 
# 
# 
# #print(logistic_ss1)
# mu_infer2 <- extract(logistic_ss2b, pars = c("l"))[[1]]
# 
# post_draws2 <- as.data.frame(t(mu_infer2))
# library(reshape2)
# post_melt2 <- melt(post_draws2)
# 
# print(logistic_ss2b)
# # Identifying the times 
# post_melt2$time <- rep(seq(1,100,1),6000)
# 
# # Plotting model fit to data with posterior draws 
# # of the latent variable 
# 
# ggplot() + geom_point(data=data.frame(x=obs_N2,y=obs_sizes2[1,]),aes(x=x,y=y),color="red") +
#   geom_point(data=data.frame(x=obs_N2,y=obs_sizes2[2,]),aes(x=x,y=y),color="red") +
#   geom_point(data=data.frame(x=obs_N2,y=obs_sizes2[3,]),aes(x=x,y=y),color="red") +
#   geom_line(data=post_melt2,aes(x=time,y=value,group=variable),size = 0.004) +
#   theme_bw()
# 
# 
# par_infer <- extract(logistic_ss2b, pars = c("r", "K", "obs_error", "proc_error"))
# 
# # r 
# r_post <- ggplot(data = data.frame(x = par_infer[[1]]),aes(x)) + stat_density(alpha = 0.5) +
#   geom_vline(aes(xintercept = r),color = "red")
# print(r_post)
# 
# # K 
# K_post <- ggplot(data = data.frame(x = par_infer[[2]]),aes(x)) + stat_density(alpha = 0.5) +
#   geom_vline(aes(xintercept = K),color = "red")
# print(K_post)
# 
# # obs error 
# obsError_post <- ggplot(data = data.frame(x = par_infer[[3]]),aes(x)) + stat_density(alpha = 0.5) +
#   geom_vline(aes(xintercept = obs_error),color = "red")
# print(obsError_post)
# 
# # proc error 
# obsError_post <- ggplot(data = data.frame(x = par_infer[[4]]),aes(x)) + stat_density(alpha = 0.5) +
#   geom_vline(aes(xintercept = 0.5),color = "red")
# print(obsError_post)
# 
# 
# 
# ### Trying again with same number of datapoints but allocated differently
# 
# # Now simulating observation error
# obs_error <- 5
# Nobs <- 9
# size_obs2 <- matrix(NA,Nobs,length(size2));
# for(i in 1:N){
#   size_obs2[,i] <- size2[i] + rnorm(Nobs,0,obs_error)
# }
# 
# 
# # Thinning observations down to 10
# obs_N2 <- sort(sample(seq(1,100,1),10))
# obs_sizes2 <- size_obs2[,obs_N2]
# 
# plot(obs_N2, obs_sizes2[1,])
# for(i in 1:Nobs){
# points(obs_N2, obs_sizes2[i,],add=T)
# }
# 
# # Putting in list for Stan
# stan_data2 <- list(N=N,KK=Nobs,M=length(obs_N2),y=obs_sizes2,times=obs_N2)
# 
# logistic_ss2c <- stan(file = "logistic2_Bayes.stan", data = stan_data2, chains = 6, iter = 2000,
#                                            seed = 12345)
# 
# print(logistic_ss2c)
# pairs(logistic_ss2c, pars = c("proc_error","obs_error"))
# pairs(logistic_ss2c, pars = c("proc_error","obs_error","r","K"))
# 
# # Still lots of sampling problems but pretty accurate parameter recovery, etc. 
# # Note that there is multimodality 
# 
# traceplot(logistic_ss2c, pars = c("proc_error","obs_error"))
# 
# ### Building YUUUGE sample size 
# 
# obs_N2 <- sort(sample(seq(1,100,1),50))
# obs_sizes2 <- size_obs2[,obs_N2]
# 
# plot(obs_N2, obs_sizes2[1,])
# for(i in 1:Nobs){
#   points(obs_N2, obs_sizes2[i,],add=T)
# }
# 
# 
# stan_data2 <- list(N=N,KK=Nobs,M=length(obs_N2),y=obs_sizes2,times=obs_N2)
# 
# logistic_ss2d <- stan(file = "logistic2_Bayes.stan", data = stan_data2, chains = 6, iter = 2000,
#                       seed = 12345)
# 
# 
# traceplot(logistic_ss2d, pars = c("proc_error","obs_error"))
# print(logistic_ss2d) 
# 
# pairs(logistic_ss2d, pars = c("r", "proc_error")) 
# # Perfect multimodality here, low r/low proc_error or high r/high proc error 
# 
# # Solutions: 1) reject chain 6, 2) improve identifiability (how?)
# 
# 
# # Running a model with plugged-in 'true' value for proc error 
# logistic_ss2e <- stan(file = "logistic3_Bayes.stan", data = stan_data2, chains = 6, iter = 2000,
#                       seed = 12345)
# 
# print(logistic_ss2e) # OF course, now everything is perfect and beautiful! 
# 
# 
# 
# 
# #### Another solution: different structure! Gamma process variance, normal observation
# # variance 
# 
# M <- 5; 
# init <- 5
# N <- 100
# size3 <- matrix(NA,M,N); 
# size3[,1] <- init; 
# dk <- 0.25
# 
# for(j in 1:M) {
#   for(i in 2:N){
#   size3[j,i] <- rgamma(1,shape=(size3[j,i-1] + r*size3[j,i-1]*(1-(size3[j,i-1]/K)))/dk,rate=1/dk);
#    }
# }
# 
# plot(size3[1,])
# for(i in 2:M){
#   points(size3[i,],col=i,add=T)
# }
# 
# # Now simulating observation error
# obs_error <- 5
# KK <- 3
# size_obs3 <- matrix(NA,KK,N);
# for(i in 1:N){
#   size_obs3[,i] <- size3[1,i] + rnorm(3,0,obs_error)
# }
# plot(size_obs3[1,])
# 
# 
# 
# 
# # Thinning observations down to 30 
# obs_N3 <- sort(sample(seq(1,100,1),30))
# obs_sizes3 <- size_obs3[,obs_N3]
# 
# plot(obs_N3, obs_sizes3[1,])
# 
# 
# stan_data3 <- list(N=N,KK=3,M=length(obs_N3),y=obs_sizes3,times=obs_N3)
# 
# logistic_ss2e <- stan(file = "logistic4_Bayes.stan", data = stan_data3, chains = 1, iter = 1000,
#                       seed = 12345)
# 
# print(logistic_ss2e) # promising 
# 
# 
# logistic_ss2eb <- stan(fit = logistic_ss2e, data = stan_data3, chains = 12, iter = 4000,
#                       seed = 12345)
# 
# 
# print(logistic_ss2eb) # KACHING! 
# 
# mu_infer3 <- extract(logistic_ss2eb, pars = c("l"))[[1]]
# 
# post_draws3 <- as.data.frame(t(mu_infer3))
# post_melt3 <- melt(post_draws3)
# 
# # Identifying the times 
# post_melt3$time <- rep(seq(1,100,1),24000)
# str(post_melt3RED)
# library(dplyr)
# 
# post_melt3RED <- post_melt3 %>% filter(as.integer(variable) %in% sample(seq(1,24000,1),5000))
# 
# # Plotting model fit to data with posterior draws 
# # of the latent variable 
# 
# ggplot() + geom_point(data=data.frame(x=obs_N3,y=obs_sizes3[1,]),aes(x=x,y=y),color="red") +
#   geom_point(data=data.frame(x=obs_N3,y=obs_sizes3[2,]),aes(x=x,y=y),color="red") +
#   geom_point(data=data.frame(x=obs_N3,y=obs_sizes3[3,]),aes(x=x,y=y),color="red") +
#   geom_line(data=post_melt3RED,aes(x=time,y=value,group=variable),size = 0.002) +
#   theme_bw()
# 
# # Evaluating against "held out" data 
# obs_N3b <- seq(1,100,1)[-obs_N3]
# obs_loo3 <- size_obs3[,obs_N3b]
# 
# ggplot() + geom_point(data=data.frame(x=obs_N3,y=obs_sizes3[1,]),aes(x=x,y=y),color="red") +
#   geom_point(data=data.frame(x=obs_N3,y=obs_sizes3[2,]),aes(x=x,y=y),color="red") +
#   geom_point(data=data.frame(x=obs_N3,y=obs_sizes3[3,]),aes(x=x,y=y),color="red") +
#   geom_line(data=post_melt3RED,aes(x=time,y=value,group=variable),size = 0.002) +
#   geom_point(data = data.frame(x = obs_N3b, y = obs_loo3[1,]),aes(x=x,y=y),color = "blue") + 
#   geom_point(data = data.frame(x = obs_N3b, y = obs_loo3[2,]),aes(x=x,y=y),color = "blue") + 
#   geom_point(data = data.frame(x = obs_N3b, y = obs_loo3[3,]),aes(x=x,y=y),color = "blue") + 
#   theme_bw()

