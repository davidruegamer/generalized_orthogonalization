source("common/functions.R")

sd_sample = 1
settings <- data.frame(p = 100, n = 1000, q = 5)
sample_fun = function(n, p) rbinom(n, 1, p)
h = plogis
g = qlogis
fam = binomial

true_coef <- rnorm(max(settings$p), sd = sd_sample) 
hidden_coef <- rnorm(max(settings$q))
Zfull <- matrix(rnorm(max(settings$n)*(max(settings$p)), sd = sd_sample),
                nrow = max(settings$n))
Xfull <- 2 * Zfull[,1:(max(settings$q))] + 
  matrix(rnorm(max(settings$n)*max(settings$q), sd = 0.1),
         nrow = max(settings$n))

### simulation
set = 1

this_p <- settings[set,]$p
this_q <- settings[set,]$q
this_n <- settings[set,]$n
ppq <- this_p + this_q

Z <- Zfull[1:this_n,]
X <- Xfull[1:this_n,]

### create response
true_add_pred <- Z[,1:this_p,drop=F]%*%true_coef[1:this_p] + 
  X[,1:this_q,drop=F]%*%hidden_coef[1:this_q]
resp <- sample_fun(this_n, h(true_add_pred))

### correct Z 
Zc <- correct_Z(thisX = X, thisZ = Z, y = resp, 
                maxiter = 10000, printevery = 1, verbose = T#,
                # lr = function(iter) if(iter<500) 0.5 else 0.9
                )

prediction_model <- glm(resp ~ -1 + Zc, family = fam)
summary(prediction_model)
yhat <- predict(prediction_model, type = "response")
summary(suppressWarnings(glm(yhat ~ -1 + X, family = fam)))

