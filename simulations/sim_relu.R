library(deepregression)

sd_sample = 1
settings <- data.frame(p = 100, n = 1000, q = 5)
sample_fun = function(n, m) rnorm(n, m, sd = 1)

pwl_gen <- function(n = 11, min_x = -4, max_x = 4, max_incr = 0.5, seed = 32){
  
  set.seed(seed)
  
  stopifnot(n%%2!=0)
  
  # Generate points for the negative x domain
  x_values <- runif(n - 1, min = min_x, max = max_x)
  x_values <- c(sort(- abs(x_values[1:(n/2)])), 0, sort(abs(x_values[(n/2+1):n])))
  min_x <- min(x_values)
  max_x <- max(x_values)
  
  incr <- runif(n - 1, min = 0, max = max_incr)
  
  left_incr <- cumsum(incr[1:(n/2)])
  right_incr <- cumsum(incr[(n/2+1):n])
  
  y_values <- c(0 - rev(left_incr), 0, right_incr)
  
  function(x){
    y <- numeric(length(x))
    
    for (i in 1:(n - 1)) {
      mask <- x >= x_values[i] & x < x_values[i + 1]
      slope <- (y_values[i + 1] - y_values[i]) / (x_values[i + 1] - x_values[i])
      y[mask] <- slope * (x[mask] - x_values[i]) + y_values[i]
    }
    
    # Handle the endpoints
    y[x <= min_x] <- y_values[1]
    y[x >= max_x] <- y_values[length(y_values)]
    
    return(y)
  }
  
}

functions <- list(
  relu = function(x) return(pmax(0, x)),
  leakyrelu = function(x) return(x * (x > 0) + 0.1 * (x < 0)),
  threesteps = function(x) return(x * (x > 0.5) + 0 * (x <= 0.5 & x > -0.5) - 0.5 * (x <= -0.5))
)

pieces <- c(3,5,7,9,15,21)

pwfs <- lapply(pieces, function(nn) list(function(x) pwl_gen(n = nn, seed = 1)(x),
                                               function(x) pwl_gen(n = nn, seed = 2)(x),
                                               function(x) pwl_gen(n = nn, seed = 3)(x),
                                               function(x) pwl_gen(n = nn, seed = 4)(x),
                                               function(x) pwl_gen(n = nn, seed = 5)(x))
)

pwfs <- unlist(pwfs, recursive = F)
names(pwfs) <- paste0("pw_", rep(pieces, each = 5), "_nr", rep(1:5))

functions <- c(functions, pwfs)

true_coef <- rnorm(max(settings$p), sd = sd_sample) 
hidden_coef <- rnorm(max(settings$q))
Zfull <- matrix(rnorm(max(settings$n)*(max(settings$p)), sd = sd_sample),
                nrow = max(settings$n))
Xfull <- 2 * Zfull[,1:(max(settings$q))] + 
  matrix(rnorm(max(settings$n)*max(settings$q), sd = 0.1),
         nrow = max(settings$n))

### helper functions
make_form <- function(respname, inds)
  as.formula(paste0(respname, " ~ ",  
                    paste(paste0("V", inds), collapse = " + "))
  )


for(i in 1:length(functions)){
  
  cat("\n#######################################\n")
  cat("\n#######################################\n")
  cat("#########", names(functions)[i], "#########\n")
  cat("#######################################\n")
  cat("#######################################\n\n")
  
  h = functions[[i]]
  
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
  y <- sample_fun(this_n, h(true_add_pred))
  
  ### orthog proj
  orthog <- function(to_orthog, orthog_with, Q = NULL)
  {
    ### construct projection matrices
    if(is.null(Q))
      Q <- qr.Q(qr(orthog_with))
    
    return(to_orthog - Q%*%(crossprod(Q, to_orthog)))
    
  }
  
  Zc <- orthog(Z, cbind(1,X))
  
  cat("Limo >>>>>>>>>>>>>>>\n\n")
  
  prediction_model <- lm(y ~ -1 + Zc)
  yhatc <- predict(prediction_model, type = "response")
  eval_model <- suppressWarnings(lm(yhatc ~ -1 + X))
  print(summary(eval_model))
  
  cat("NonLimo >>>>>>>>>>>>>>>\n\n")
  loss_function <- function(beta, x, y) {
    predictions = h(x %*% beta)
    mse = mean((y - predictions)^2)
    return(mse)
  }
  result = optim(rnorm(length(true_coef)), loss_function, x = Zc, y = y, method = "BFGS")
  yhatc <- Zc%*%result$par
  cat("Normal Init:\n")
  eval_model = optim(rnorm(length(hidden_coef)), loss_function, x = X, y = yhatc, method = "BFGS")
  cat("Coefs: ", paste(round(eval_model$par,3), collapse = " "), "; Optval: ", eval_model$value, "\n")
  cat("0 Init:\n")
  eval_model = optim(rep(0,length(hidden_coef)), loss_function, x = X, y = yhatc, method = "BFGS")
  cat("Coefs: ", paste(round(eval_model$par,3), collapse = " "), "; Optval: ", eval_model$value, "\n")
  
  cat("\nGD >>>>>>>>>>>>>>>\n\n")
  df <- function(x) x %>% layer_dense(units = 1, activation = "relu")
  prediction_model <- deepregression(y = y, 
                        list_of_formulas = 
                          list(as.formula(
                            paste0("~ -1 + d(", 
                                   paste(paste0("V", 1:this_p), 
                                         collapse = ", "),")")), ~1),
                        data = as.data.frame(Zc),
                        list_of_deep_models = list(d=df))
  prediction_model %>% fit(epochs = 1000, verbose = F, patience = 50, 
              early_stopping = TRUE)
  yhatc <- predict(prediction_model)
  eval_model <- deepregression(y = yhatc, 
                        list_of_formulas = 
                          list(as.formula(
                            paste0("~ -1 + d(", 
                                   paste(paste0("V", 1:this_q), 
                                         collapse = ", "),")")), ~1),
                        data = as.data.frame(X),
                        list_of_deep_models = list(d=df))
  eval_model %>% fit(epochs = 1000, verbose = F, patience = 50, 
                     early_stopping = TRUE)
  cat("Coefs: ", paste(round(as.matrix(eval_model$model$weights[[1]])[,1],3), collapse = " "))
  
}
  