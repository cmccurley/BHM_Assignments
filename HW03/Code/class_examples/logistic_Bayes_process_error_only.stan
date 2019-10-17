data{
  int<lower = 0> N; 
  real y[N];
}

parameters{
  real<lower = 0> r; 
  real<lower = 0> K; 
  real<lower = 0> proc_error;
 
}

model{
  
  // obs_error ~ normal(0,10); 
  K ~ normal(0,100); 
  r ~ normal(0, 1); 
  
  proc_error ~ normal(0,3); 
  
 // Note that we are constraining latent variable 
 // with observations at specific times 
  for(i in 2:N)
  y[i] ~ normal(y[i-1] + r*y[i-1]*(1 - y[i-1]/K), proc_error);
}
