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

# Putting in list for Stan
stan_data <- list(N=N,y=y)


# Fitting model

library(rstan)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)
Sys.setenv(LOCAL_CPPFLAGS = '-march=native')

initf1 <- function() {
  list(r = 0.2, K = 50, proc_error = 2)
}


logistic_ss1 <- stan(file = "logistic_Bayes_process_error_only.stan", data = stan_data,
                     chains = 6, iter = 1000, init = initf1)


#print(logistic_ss1)
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
# 
# # proc error
# procError_post <- ggplot(data = data.frame(x = par_infer[[4]]),aes(x)) + stat_density(alpha = 0.5) +
#   geom_vline(aes(xintercept = obs_error),color = "red")
# print(procError_post)