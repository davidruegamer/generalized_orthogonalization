wrapper_orthog <- function(
    response, # y
    predictors, # Z
    sensitive, # X
    unfairness = NULL, # not used
    family,
    maxdiff = 1e-7
)
{

  source("../../simulations/common/functions.R")
  
  modfun <- switch(family,
                   gaussian = function(y,x) glmgen(y, x, family, TRUE, FALSE),
                   binomial = function(y,x) glmgen(y, x, family, TRUE),
                   poisson = function(y,x) glmgen(y, x, family, TRUE),
                   multinomial = function(y, x) nnet::multinom(y ~ -1 + x))
  
  predfun <- switch(family,
                    gaussian = predict,
                    binomial = function(mod) predict(mod, type = "response"),
                    poisson = function(mod) predict(mod, type = "response"),
                    multinomial = predict)
  
  Zc <- if(family == "gaussian") orthog(to_orthog = predictors, 
                                        orthog_with = sensitive) else
                                          correct_Z(sensitive, predictors, 
                                                    maxdiff = maxdiff, 
                                                    response, family, verbose = T,
                                                    with_intercept = TRUE) 
  
  prediction_mod <- modfun(response, Zc)
  yhat <- predfun(prediction_mod)
  evaluation_mod <- modfun(yhat, sensitive)
  
  return(list(pred_mod = prediction_mod,
              yhat = yhat,
              eval_mod = evaluation_mod))

}