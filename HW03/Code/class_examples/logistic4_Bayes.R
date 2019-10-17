
N <- 100 # Length of simulation 
r <- 0.1 # intrinsic growth rate
K <- 100 # carrying capacity 
init <- 5 # initial condition 
M = N
KK = 5
k =0.2

nEachStep =  4
z = rep(NA, N);
z[1] <- init; 
for(i in 2:N){
  z[i]  = rgamma(1,(z[i-1] + r*z[i-1]*(1-z[i-1]/K))/k,(1/k));
}
# plot(z)
# Now simulating observation error
obs_error <- 5

y = matrix(NA, nEachStep, N)


for(i in 1:N){
  for(j in 1:nEachStep){
    y[j,i] <- z[i] + rnorm(1,0,obs_error)
  }
}




# Putting in list for Stan
# stan_data <- list(N=N,M=length(obs_N),y=,times=obs_N)


# Fitting model 

# library(rstan)
# options(mc.cores = parallel::detectCores())
# rstan_options(auto_write = TRUE)
# Sys.setenv(LOCAL_CPPFLAGS = '-march=native')
# 
# initf1 <- function() {
#   list(r = 0.2, K = 50, mu = array(1,dim=c(1,100)), obs_error = 1)
# } 
# 
# 
# logistic_ss1 <- stan(file = "logistic_Bayes.stan", data = stan_data,
#                      chains = 6, iter = 1000, init = initf1)