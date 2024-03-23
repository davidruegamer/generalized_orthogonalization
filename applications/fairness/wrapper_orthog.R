wrapper_orthog <- function(
    response, # y
    predictors, # Z
    sensitive, # X
    unfairness = NULL, # not used
    family,
    maxdiff = 1e-7,
    correct_method = c("lagrangian", "project")
)
{
  
  correct_method <- match.arg(correct_method)

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
  
  if(family == "gaussian"){ 
    
    yhat <- predict(modfun(response, orthog(to_orthog = predictors, orthog_with = sensitive)))
    
  }else{
    
    thisZ <- model.matrix(~ ., data = predictors)
    thisX <- model.matrix(~ ., data = sensitive)
    y <- if(family == "binomial") as.numeric(response)-1 else response
    if(correct_method == "project"){
      yhat <- correct_Z(thisX, thisZ, y, family, what = "pred") 
    }else{
      yhat <- lagrangianConstr(thisX, thisZ, y, family, what = "pred")
    }
    
    if(family=="binomial"){ 
      cat("ACC w/o corr.: ", Metrics::accuracy(y, predfun(modfun(response, thisZ))>0.5), "\n",
          "ACC w/ corr.: ", Metrics::accuracy(y, yhat>0.5), "\n")
    }else{
      cat("MSE w/o corr.: ", Metrics::rmse(y, predfun(modfun(response, thisZ))), "\n",
          "MSE w/ corr.: ", Metrics::rmse(y, yhat))
    }
    
    
  }
  # browser()
  evaluation_mod <- modfun(yhat, sensitive)
  
  return(evaluation_mod)

}