data{
  int<lower = 0> N; 
  int<lower = 0> M;
  int<lower=0> KK; 
  real y[KK,M];
  int<lower = 0> times[M]; 
}

parameters{
  real<lower = 0> r; 
  real<lower = 0> K; 
  real<lower = 0> proc_error; 
  real<lower = 0> obs_error;
  real<lower = 0> mu_init; 
  real<lower = 0> l[N]; 
}


model{
  real mu[N]; 
  
  obs_error ~ normal(0,10); 
  proc_error ~ normal(0,1); 
  K ~ normal(0,100); 
  r ~ normal(0, 1); 
  // kludgy fix for now
  mu_init ~ normal(1,1); 
  
  // tricky way of iterating through latent variable 
  l[1] ~ normal(mu_init,proc_error); 
  mu[1] = mu_init; 
  for(i in 2:N){
    mu[i] = l[i-1] + r*l[i-1]*(1 - l[i-1]/K);
    l[i] ~ normal(mu[i],proc_error);
  }
  
  
  // Note that we are constraining latent variable 
  // with observations at specific times 
  
  for(j in 1:M)
    y[,j] ~ normal(mu[times[j]],obs_error);
}