data{

  int<lower = 0> N;

  int<lower = 0> M;

  int<lower=0> KK;

  int<lower = 0> times[KK];

  real y[M,KK];

 

}

 

parameters{

  real<lower = 0> r;

  real<lower = 0> K;

  real<lower = 0, upper = 2> k;

  real<lower = 0> obs_error;

  vector<lower = 0>[M] mu_init;

  real<lower = 0> l[M,N];

 

}

 

 

model{

  real mu[M,N];

 

   obs_error ~ normal(0,10);

   K ~ normal(0,100);

   r ~ normal(0, 1);

   k ~ normal(0, 1);

  // kludgy fix for now

  mu_init ~ normal(5,5);

  

  // tricky way of iterating through latent variable

// initial state

  for(i in 1:M){

  l[i,1] ~ gamma(mu_init[i]/k,1/k);

  mu[i,1] = mu_init[i];

  }

// rest of states

for(j in 1:M){

  for(i in 2:N){

    mu[j,i] = l[j,i-1] + r*l[j,i-1]*(1 - l[j,i-1]/K);

    l[j,i] ~ gamma(mu[j,i]/k,1/k);

  }

}

 

  // Note that we are constraining latent variable

  // with observations at specific times

  

  for(j in 1:KK){

    y[,j] ~ normal(mu[,times[j]],obs_error);

  }

}