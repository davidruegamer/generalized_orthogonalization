library(dplyr)

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

### get GLM derivative
get_derivGLM <- function(mod){
  
  if(mod$family$family=="poisson")
    return(1/mod$fitted.values)
  if(mod$family$family=="binomial")
    return(1/(mod$fitted.values*(1-mod$fitted.values)))
  
}

### apply weight augmentation
wAug <- function(X, W) sweep(X, 1, W, "*")

### formula handling
form_from_df <- function(mat, int, ofsf = "", outc = "y")
{
  as.formula(paste0(outc, " ~ ", int, 
                    paste(colnames(mat), 
                          collapse = " + "), ofsf)
  )
}

### helper functions
make_form <- function(respname, inds)
  as.formula(paste0(respname, " ~ ",  
                    paste(paste0("V", inds), collapse = " + "))
  )

### glm generic
glmgen <- function(y, mat_or_df, fam, # ofs = NULL, 
                   intercept = TRUE, g = TRUE)
{
  
  fun <- if(g) glm else function(..., fam) lm(...)
  
  if(is.data.frame(mat_or_df)){
    
    # ofsf <- if(!is.null(ofs)) "+ ofs" else ""
    int <- if(intercept) "1 + " else "-1 + "
    form <- form_from_df(mat_or_df, int) #, ofsf)
    
    fun(form, family = fam, data = cbind(y = y, mat_or_df))
    
  }else if(is.matrix(mat_or_df)){
    
    if(intercept & !sum(mat_or_df[,1]==1)==nrow(mat_or_df)){ 
      
      # if(!is.null(ofs))
        # fun(y ~ 1 + mat_or_df + offset(ofs), family = fam) else
          fun(y ~ 1 + mat_or_df, family = fam)
      
    }else{
        
      # if(!is.null(ofs))
        # fun(y ~ -1 + mat_or_df + offset(ofs), family = fam) else
          fun(y ~ -1 + mat_or_df, family = fam)
      
    }
    
  }else{
    
    stop("Wrong format.")
    
  }
  
  
}

fisher_scoring_glm_corr <- function(Q, Z, y, distribution = "binomial", 
                                    max_iter = 10000, conv_threshold = 1e-3,
                                    return_what = c("gamma", "Z", "pred"),
                                    lr = 0.5, fac_improvem = 100, onlyimprov = TRUE,
                                    gamma_start = rep(0, ncol(Z))) {
  gamma <- gamma_start
  gamma_incr_old <- rep(100, ncol(Z))
  return_what <- match.arg(return_what)
  # gamma[1] <- if(distribution == "binomial") qlogis(mean(y)) else log(mean(y))
  
  for (i in 1:max_iter) {
    
    eta <- Z %*% gamma
    eta <- eta - Q %*% crossprod(Q, eta)
    
    if (distribution == "binomial") {
      mu <- exp(eta) / (1 + exp(eta))
      g_deriv <- dlogis(eta)
      V_mu <- mu * (1-mu)
      h <- plogis
    } else if (distribution == "poisson") {
      mu <- exp(eta)
      g_deriv <- 1/mu
      V_mu <- mu
      h <- exp
    } else {
      stop("Unsupported distribution")
    }
    
    W <- 1/(g_deriv^2 * V_mu)
    # W <- W/min(W)
    G_Z <- g_deriv
    
    Zcurrent <- Z #cbind(1, wAug(PXbot, (1/W)*(1/G_Z))%*%wAug(Z[,-1], G_Z*W))

    r <- eta + ((y - mu) - Q %*% crossprod(Q, (y - mu))) * G_Z
    gamma_new <- coef(lm(r ~ -1 + Zcurrent, weights = W[,1]))
    gamma_new[is.na(gamma_new)] <- 0

    gamma_incr <- gamma - gamma_new
    
    if (max(abs(gamma_incr)) < conv_threshold | 
        max(abs(gamma_incr - gamma_incr_old)) < conv_threshold/fac_improvem
        ) {
      break
    }
    cat("Iteration: ", i, " -- ", max(abs(gamma_incr)), "\n")
    if(onlyimprov & max(abs(gamma_incr)) - max(abs(gamma_incr_old)) > 0)
    {
      if(return_what=="gamma")
        return(gamma)
      if(return_what=="Z")
        return(Zcurrent)
      return(h(Zcurrent%*%gamma))
    }
    gamma <- lr * gamma_new + (1-lr) * gamma
    gamma_incr_old <- gamma_incr
    if(i == max_iter)
      warning("GLM not converged")
  }
  
  if(return_what=="gamma")
    return(gamma)
  if(return_what=="Z")
    return(Zcurrent)
  return(h(Zcurrent%*%gamma))
}


### Function implementing Algorithm 1
correct_Z <- function(thisX, thisZ, y, fam, what = "Z", ...)
{
  

  if(class(fam)=="family")
    fam <- fam$family
  
  Q <- qr.Q(qr(thisX))
  # PXbot <- diag(rep(1, nrow(thisX))) - tcrossprod(Q)
  
  ### Compute Z^c
  fisher_scoring_glm_corr(Q, thisZ, y, distribution = fam, 
                          return_what = what, ...)
  
}

lagrangianConstr <- function(thisX, thisZ, y, fam, what = "gamma", 
                             startval_lambda = 1, ...)
{
  
  if(class(fam)=="family")
    fam <- fam$family
  
  centeredX <- scale(thisX, scale=F)
  
  if (fam == "binomial") {
    b <- function(theta) log(1 + exp(theta))
    theta <- function(mu) log(mu / (1-mu))
    mu <- function(gamma) exp(thisZ%*%gamma) / (1 + exp(thisZ%*%gamma))
    derivmu <- function(gamma) mu(gamma) * (1-mu(gamma))
  } else if (fam == "poisson") {
    b <- function(theta) exp(theta)
    theta <- function(mu) log(mu)
    mu <- function(gamma) exp(thisZ%*%gamma)
    derivmu <- function(gamma) mu(gamma) 
  } else {
    stop("Unsupported distribution")
  }
  loglik <- function(gamma) sum(y*theta(mu(gamma)) - b(theta(mu(gamma))))
  score <- function(gamma) crossprod(thisZ, (y-mu(gamma)))
  inf <- function(gamma) c(crossprod(t(centeredX)%*%mu(gamma)))
  deriv_inf <- function(gamma) 2*t(crossprod((centeredX*sqrt(derivmu(gamma)[,1])),
                                             thisZ*sqrt(derivmu(gamma)[,1]))
                                   )%*%(t(centeredX)%*%mu(gamma))
  damp_term <- function(gamma) inf(gamma)^2 / 2
  obj <- function(params) -loglik(params[-1]) + damp_term(params[-1]) + 
    exp(params[1])*inf(params[-1])
  obj_g <- function(params) c(
    exp(params[1])*inf(params[-1]),
    - score(params[-1]) +
      params[1]*deriv_inf(params[-1])+
      deriv_inf(params[-1])*inf(params[-1])
  )
  opt <- optim(c(startval_lambda, rep(0,ncol(thisZ))), 
               obj, control = list(maxit = 10000), 
               gr = obj_g,
               method = "BFGS"#, 
               # lower = c(0, rep(-100, ncol(thisZ)))
               )
  if(what == "gamma") return(opt$par[-1])
  if(what == "pred") return(mu(opt$par[-1]))
  if(what == "lambda") return(opt$par[1])
  if(what == "params") return(opt$par)
  return(opt)
  
}


# Function for simulation
sim_function <- function(sample_fun, h, g, fam,
                         sd_sample = 0.1,
                         q_vals = c(10, 50, 100),
                         p_vals = c(2, 5, 10),
                         n_vals = c(200, 1000, 5000, 10000),
                         rep_vals = 1:10,
                         load_fac_X_on_Z = c(0, 1 ,2),
                         nrcores = 4,
                         which_corr = c("iterProj", "Lagrangian")
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
    
    X <- scale(X, scale = FALSE)
    
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
    yhat_score <- matrix(predict(prediction_model, type = "link"))
    yhat_prob <- matrix(predict(prediction_model, type = "response"))
    
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
    # browser()
    yhatc_score <- orthog(to_orthog = yhat_score, orthog_with = X)
    if(which_corr == "iterProj"){
      yhatc_prob <- correct_Z(X, Z, resp, fam, what = "pred")
    }else if(which_corr == "Lagrangian"){
      yhatc_prob <- lagrangianConstr(X, Z, resp, fam, what = "pred")
    }else{
      stop("Not implemented.")
    }
    
    # colnames(Zc_prob) <- paste0("V", 1:ncol(Zc_prob))
    # yhatc_prob <- predict(glm(make_form("y", 1:this_q),
    #                           data = cbind(y = resp, 
    #                                        as.data.frame(Zc_prob)), 
    #                           family = fam))
    
    ### check again using evaluation model
    linear_proj_model <- lm(make_form("yhat", 1:this_p), 
                            data = cbind(yhat = yhatc_score,
                                         as.data.frame(X)))
    corr_nl_l <- summary(linear_proj_model)$coefficients[2:(this_p+1),c(1,4)]

    nonlinear_proj_model <- glm(make_form("yhat", 1:this_p),
                                data = cbind(yhat = (yhatc_prob),
                                             as.data.frame(X)),
                                family = fam)
    corr_nl_nl <- summary(nonlinear_proj_model)$coefficients[2:(this_p+1),c(1,4)]
    
    return(list(res=rbind(cbind(uncorr_nl_l, type = "NL/L", corr = 0),
                          cbind(uncorr_nl_nl, type = "NL/NL", corr = 0),
                          cbind(corr_nl_l, type = "NL/L", corr = 1),
                          cbind(corr_nl_nl, type = "NL/NL", corr = 1)),
                setting=settings[set,]))
           
  }, mc.cores = nrcores)
  
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
    facet_grid(name ~ n*q, scales="free", 
               labeller = labeller(n = label_both, q = label_both)) + 
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