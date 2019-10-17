data {
  int<lower=0> J;         // number of schools 
  real y[J];              // estimated treatment effects
  real<lower=0> sigma[J]; // standard error of effect estimates 
  real a;                 // mean value for prior on mu
  real<lower=0> b;        // std for prior on mu
  real c;                 // mean value for prior on tau
  real<lower=0> d;        // std for prior on tau
}
parameters {
  real mu;                // population treatment effect
  real<lower=0> tau;      // standard deviation in treatment effects
  vector[J] eta;          // unscaled deviation from mu by school
}
transformed parameters {
  vector[J] theta = mu + tau * eta;        // school treatment effects
}
model {
  target += normal_lpdf(eta | 0, 1);       // prior log-density
  target += normal_lpdf(y | theta, sigma); // log-likelihood
  target += normal_lpdf(mu | a, b);
  target += normal_lpdf(tau | c, d);
}