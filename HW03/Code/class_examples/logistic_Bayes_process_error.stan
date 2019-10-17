data{
  int<lower = 0> N; 
  int<lower = 0> M; 
  real y[M];
  //real<lower = 0> y[N,M];
  int<lower = 0> times[M]; 
  real z[M];
}

parameters{
  real<lower = 0> r; 
  real<lower = 0> K; 
  //real<lower = 0> obs_error;
  real<lower = 0> mu_init;
  real<lower = 0> proc_error;
}

transformed parameters{
// latent variable mu 
  real<lower=0> mu[N];
  //real eta[N];
  
  mu[1] = mu_init;
   for(i in 2:N){
    mu[i] = mu[i-1] + r*mu[i-1]*(1 - mu[i-1]/K);
   }
}

model{
  
  //obs_error ~ normal(0,10); 
  K ~ normal(0,100); 
  r ~ normal(0, 1); 
  proc_error ~ normal(0,2);
  
  //mu_init ~ normal(y[1],obs_error); 
 // Note that we are constraining latent variable 
 // with observations at specific times 
  for(j in 2:M)
  y[j] ~ normal(mu[times[j]],proc_error);

}