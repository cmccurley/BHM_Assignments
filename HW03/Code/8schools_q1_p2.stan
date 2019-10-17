// saved as 8schools.stan
data {
  int<lower=0> J;         // number of schools 
  real y[J];              // estimated treatment effects
  real<lower=0> sigma[J]; // standard error of effect estimates 
  real<lower=0> scale;
}
parameters {
  real mu;                // population treatment effect
  vector[J] eta;          // unscaled deviation from mu by school
  real<lower=0> tau;      // standard deviation in treatment effects
}
transformed parameters {
  vector[J] theta = mu + tau * eta;        // school treatment effects
}
model {
  tau ~ normal(0, scale);
  target += normal_lpdf(eta | 0, 1);       // prior log-density
  target += normal_lpdf(y | theta, sigma); // log-likelihood
}
