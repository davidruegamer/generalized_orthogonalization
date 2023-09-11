### Function computing the orthogonal projection
orthog <- function(to_orthog, orthog_with)
{
  
  if(!is.matrix(to_orthog)) 
    to_orthog <- model.matrix(form_from_df(to_orthog,
                                           int="1 + ",outc=""),
                              data = to_orthog)[,-1]
  if(!is.matrix(orthog_with)) 
    orthog_with <- model.matrix(form_from_df(orthog_with,
                                             int="1 + ",outc=""),
                                data = orthog_with)[,-1]
  
  ### construct projection matrices
  Qtilde <- qr.Q(qr(orthog_with))
  
  return(to_orthog - Qtilde%*%(crossprod(Qtilde, to_orthog)))
  
}

### get GLM weights^(1/2)
get_sqrW <- function(mod) sqrt(mod$weights)

### apply weight augmentation
wAug <- function(X, W) t(t(X) * W)

### formula handling
form_from_df <- function(mat, int, outc = "y")
{
  as.formula(paste0(outc, " ~ ", int, 
                    paste(colnames(mat), 
                          collapse = " + "))
  )
}

### glm generic
glmgen <- function(y, mat_or_df, fam, intercept, g = TRUE)
{
  
  fun <- if(g) glm else function(..., fam) lm(...)
  
  if(is.data.frame(mat_or_df)){
    
    int <- if(intercept) "1 + " else "-1 + "
    form <- form_from_df(mat_or_df, int)
    
    fun(form, family = fam, data = cbind(y = y, mat_or_df))
    
  }else if(is.matrix(mat_or_df)){
    
    if(intercept & !sum(mat_or_df[,1]==1)==nrow(mat_or_df)){ 
      
      fun(y ~ 1 + mat_or_df, family = fam) 
      
    }else{
        
      fun(y ~ -1 + mat_or_df, family = fam)
      
    }
    
  }else{
    
    stop("Wrong format.")
    
  }
  
  
}

### Function implementing Algorithm 1
correct_Z <- function(thisX, thisZ, y, fam, maxdiff = 1e-7, verbose = FALSE, 
                      printevery = 10, lr = function(iter) 1, maxiter = 1000,
                      return_hist = FALSE, with_intercept = NULL, scalefun = scale,
                      with_intercept_x = TRUE, with_intercept_z = TRUE)
{
  
  dontstop <- TRUE
  iter <- 0
  if(!verbose) printevery <- Inf
  if(!is.null(with_intercept)){
    with_intercept_x <- with_intercept
    with_intercept_z <- with_intercept
  }
  beta <- rep(0, ncol(thisX) + with_intercept_x)
  if(return_hist) hist <- c()
  
  while(TRUE){
    
    prediction_model <- glmgen(y, thisZ, fam, with_intercept_z) 
    yhat <- predict(prediction_model, type = "response")
    eval_model <- suppressWarnings(glmgen(yhat, thisX, fam, with_intercept_x))

    ### extract important terms
    Ups_sqrt <- get_sqrW(eval_model)
    Psi_sqrt <- get_sqrW(prediction_model)
    Xtilde <- wAug(model.matrix(eval_model)[,-with_intercept_x], Ups_sqrt)
    Ztilde <- wAug(model.matrix(prediction_model)[,-with_intercept_z], Psi_sqrt)
    
    ### compute Z^c
    Zc <- scalefun(orthog(Ztilde, Xtilde))
    
    ### Check convergence
    thiscrit <- crossprod(beta - coef(eval_model))
    if(return_hist) hist <- c(hist, thiscrit)
    if(iter%%printevery==0 & printevery!=Inf) cat("Iter: ", iter, "; Max difference:", thiscrit, "\n")
    iter <- iter + 1
    beta <- coef(eval_model)
    
    if(thiscrit < maxdiff){ 
      if(return_hist) attr(Zc, "hist") <- hist
      return(Zc) 
    }else if(thiscrit > 10e10){
      stop("Convergence failed with no solution.")
    }else if(iter == maxiter){
      warning("Not converged.")
      if(return_hist) attr(Zc, "hist") <- hist
      return(Zc)
    }else{
      thisZ <- lr(iter)*Zc + (1-lr(iter))*model.matrix(prediction_model)[,-with_intercept_z]
    }
    
  }
  
}

# Function for simulation
sim_function <- function(sample_fun, h, g, fam,
                         sd_sample = 0.1,
                         q_vals = c(10, 50, 100),
                         p_vals = c(2, 5, 10),
                         n_vals = c(200, 1000, 5000, 10000),
                         rep_vals = 1:10,
                         load_fac_X_on_Z = c(0, 1 ,2)
                         )
{
  
  library(dplyr)
  library(parallel)
  
  ### settings
  settings <- expand.grid(
    p = p_vals,
    q = q_vals,
    n = n_vals,
    reps = rep_vals,
    load = load_fac_X_on_Z
  )
  
  ### data generation
  set.seed(42)
  true_coef <- rnorm(max(settings$q), sd = sd_sample) 
  hidden_coef <- rnorm(max(settings$p))
  Zfull <- matrix(rnorm(max(settings$n)*(max(settings$q)), sd = sd_sample),
                  nrow = max(settings$n))
  Xfull_wo <- matrix(rnorm(max(settings$n)*max(settings$p), sd = 0.1),
           nrow = max(settings$n))
  
  ### simulation
  res <- mclapply(1:nrow(settings), function(set){
    
    this_p <- settings[set,]$p
    this_q <- settings[set,]$q
    this_n <- settings[set,]$n
    load   <- settings[set,]$load
    ppq <- this_p + this_q
    
    X <- Xfull_wo[1:this_n,1:this_p] + load * Zfull[1:this_n,1:this_p]
    Z <- Zfull[1:this_n,]
    
    ### create response
    true_add_pred <- Z[,1:this_q,drop=F]%*%true_coef[1:this_q] + 
      X[,1:this_p,drop=F]%*%hidden_coef[1:this_p]
    set.seed(set)
    resp <- sample_fun(this_n, h(true_add_pred))
    
    ### define and fit prediction model
    prediction_model <- glm(make_form("y", 1:this_q),
                            data = cbind(y = resp, 
                                         as.data.frame(Z)), 
                            family = fam)
    
    ### get predicted values as scores and probabilities
    yhat_score <- predict(prediction_model, type = "link")
    yhat_prob <- predict(prediction_model, type = "response")
    
    ### define and fit evaluation models
    linear_proj_model <- lm(make_form("yhat", 1:this_p), 
                            data = cbind(yhat = yhat_score,
                                         as.data.frame(X)))
    uncorr_nl_l <- summary(linear_proj_model)$coefficients[2:(this_p+1),c(1,4)] 

    nonlinear_proj_model <- glm(make_form("yhat", 1:this_p),
                                data = cbind(yhat = yhat_prob,
                                             as.data.frame(X)),
                                family = fam)
    uncorr_nl_nl <- summary(nonlinear_proj_model)$coefficients[2:(this_p+1),c(1,4)] 
    
    ### correct model
    yhatc_score <- orthog(to_orthog = yhat_score, orthog_with = X)
    Zc_prob <- correct_Z(X, Z, resp, fam, return_hist = T, with_intercept = F)
    yhatc_prob <- predict(glm(make_form("y", 1:this_q),
                              data = cbind(y = resp, 
                                           as.data.frame(Zc_prob)), 
                              family = fam))
    conv_hist <- attr(Zc_prob, "hist")
    
    ### check again using evaluation model
    linear_proj_model <- lm(make_form("yhat", 1:this_p), 
                            data = cbind(yhat = yhatc_score,
                                         as.data.frame(X)))
    corr_nl_l <- summary(linear_proj_model)$coefficients[2:(this_p+1),c(1,4)]

    nonlinear_proj_model <- glm(make_form("yhat", 1:this_p),
                                data = cbind(yhat = h(yhatc_prob),
                                             as.data.frame(X)),
                                family = fam)
    corr_nl_nl <- summary(nonlinear_proj_model)$coefficients[2:(this_p+1),c(1,4)]
    
    return(list(res=rbind(cbind(uncorr_nl_l, type = "NL/L", corr = 0),
                          cbind(uncorr_nl_nl, type = "NL/NL", corr = 0),
                          cbind(corr_nl_l, type = "NL/L", corr = 1),
                          cbind(corr_nl_nl, type = "NL/NL", corr = 1)),
                setting=settings[set,],
                hist = conv_hist))
           
  }, mc.cores = 8)
  
}

# Function to plot results
plot_function <- function(res)
{
  
  library(ggplot2)
  library(viridis)
  library(tidyr)
  
  ress <- do.call("rbind", lapply(res, function(r){
    # if(ncol(res[[set]])==3)
    #   ret <- cbind(var = 1:nrow(res[[set]]), res[[set]][,1], NA, res[[set]][,2:3], 
    #                settings[set,]) else
    #     ret <- cbind(var = 1:nrow(res[[set]]), res[[set]], settings[set,])
    #   colnames(ret) <- c("var", "coef", "pval", "type", "corr", colnames(settings))
    #   return(ret)
    cbind(as.data.frame(r$res), r$setting)
  }))
  
  ress[,c(1:2,4)] <- lapply(ress[,c(1:2,4)], as.numeric)
  colnames(ress)[1:2] <- c("coef", "pval")
  
  my_palette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", 
                  "#0072B2", "#D55E00", "#CC79A7")
  
  lapply(unique(ress$load), function(ll) ggplot(ress %>% pivot_longer(coef:pval) %>% 
           mutate(
             corr = factor(corr, levels = 0:1, 
                           labels = c("w/o correction",
                                      "w/  correction")),
             name = factor(name, levels = c("coef", "pval"),
                           labels = c("effect", "p-value")),
             type = factor(type, levels = unique(type),
                           labels = c(expression(C^l), expression(C^h))
             )
           ) %>% filter(abs(value) <= 10) %>% filter(load == ll),  
         aes(x = type, y=value, colour=corr)) +
    geom_boxplot() +
    facet_grid(name ~ n*q, scales="free", labeller = labeller(n = label_both, q = label_both)) + 
    theme_bw() + 
    scale_colour_manual(values = my_palette) + 
    theme(
      legend.title = element_blank(),
      text = element_text(size = 14),
      legend.position="bottom"
    ) + 
    scale_x_discrete(labels = c(expression(italic(C)^l), expression(italic(C)^h)))
  )
  
}

### helper functions
make_form <- function(respname, inds)
  as.formula(paste0(respname, " ~ ",  
                    paste(paste0("V", inds), collapse = " + "))
  )