data{
  int<lower = 0> N; 
  int<lower = 0> M; 
  real y[M];
  int<lower = 0> times[M]; 
}

parameters{
  real<lower = 0> r; 
  real<lower = 0> K; 
  real<lower = 0> obs_error;
  real<lower = 0> mu_init; 
}

transformed parameters{
# latent variable mu 
  real<lower=0> mu[N]; 
    mu[1] = mu_init; 
   for(i in 2:N){
    mu[i] = mu[i-1] + r*mu[i-1]*(1 - mu[i-1]/K);
   }
}

model{
  
  // obs_error ~ normal(0,10); 
  K ~ normal(0,100); 
  r ~ normal(0, 1); 
  
  obs_error ~ normal(0,10); 
  
  mu_init ~ normal(y[1],obs_error); 
 # Note that we are constraining latent variable 
 # with observations at specific times 
  for(j in 2:M)
  y[j] ~ normal(mu[times[j]],obs_error);
}