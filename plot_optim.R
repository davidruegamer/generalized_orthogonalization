library(ggplot2)
library(dplyr)
library(gridExtra)

source("simulations/common/functions.R")
fisher_scoring_glm_corr <- function(Q=NULL, Z, y, lr = 1, max_iter = 110, gamma_start = rep(0, ncol(Z))) {
  
  gamma <- gamma_start
  gamma_list <- list()
  gamma_list[[1]] <- gamma
  gamma_incr_old <- rep(100, ncol(Z))
  
  for (i in 1:max_iter) {
    
    eta <- Z %*% gamma
    if(!is.null(Q)) eta <- eta - Q %*% crossprod(Q, eta)
    
    mu <- exp(eta) / (1 + exp(eta))
    g_deriv <- dlogis(eta)
    V_mu <- mu * (1-mu)
    h <- plogis
    
    W <- 1/(g_deriv^2 * V_mu)
    G_Z <- g_deriv
    
    res <- y-mu
    if(!is.null(Q)) res <- res - Q %*% crossprod(Q,res)
    r <- eta + res * G_Z
    gamma_new <- coef(lm(r ~ -1 + Z, weights = W[,1]))
    gamma_new[is.na(gamma_new)] <- 0
    
    gamma <- lr * gamma_new + (1-lr) * gamma
    gamma_list[[i+1]] <- gamma
    
  }
  
  return(gamma_list)
    
}

# Data Generation
set.seed(133)  # For reproducibility
n_samples <- 1000
z1 <- rnorm(n_samples)
z2 <- rnorm(n_samples)
Z <- cbind(z1, z2)
Z_with_intercept <- cbind(rep(1, n_samples), Z)  # Adding intercept
true_gammas <- c(1, 0.5, -1)  # Intercept, beta for z1, and beta for z2
lin_comb <- Z_with_intercept %*% true_gammas
p <- plogis(lin_comb)
y <- rbinom(n_samples, size = 1, prob = p)
eps <- rnorm(n_samples)
x <- plogis(z1 + eps)
Q <- qr.Q(qr(matrix(x)))
Zcor <- Z_with_intercept - Q%*%crossprod(Q,Z_with_intercept)

params <- lagrangianConstr(matrix(x), Z, y, "binomial", what = "params")
lambda <- exp(params[1])

# Define the loss function for logistic regression
logistic_loss <- function(beta, X, y) {
  probabilities <- plogis(X %*% beta)
  -sum(y * log(probabilities) + (1 - y) * log(1 - probabilities))
}

# Define the loss function for logistic regression (classical orthog)
logistic_loss_l <- function(beta, Xc, y) {
  probabilities <- plogis(Xc %*% beta)
  -sum(y * log(probabilities) + (1 - y) * log(1 - probabilities))
}

# Define penalty
pen_loss <- function(gamma, Z, X, y) {
  probabilities <- plogis(Z %*% gamma)
  -sum(y * log(probabilities) + (1 - y) * log(1 - probabilities)) + 
    lambda*c(crossprod(t(scale(X, scale=F))%*%probabilities))
}

# Custom Newton method with projection of residuals
path_custom_newton <- fisher_scoring_glm_corr(Q, Z_with_intercept, y)
path_custom_newton_final <- fisher_scoring_glm_corr(Q, Z_with_intercept, y, max_iter = 400)

# Old method
path_customold_newton <- fisher_scoring_glm_corr(Q=NULL, Zcor, y)
path_customold_newton_final <- fisher_scoring_glm_corr(Q=NULL, Zcor, y, max_iter = 400)

# Standard Newton method
path_standard_newton <- fisher_scoring_glm_corr(Q=NULL, Z_with_intercept, y, max_iter = 300)

# Calculate loss surface
gamma1_range <- seq(-0.2, 1.2, length.out = 100)
gamma2_range <- seq(-1.7, 0.2, length.out = 100)
loss_surface <- outer(gamma1_range, gamma2_range, 
                      Vectorize(function(b1, b2) logistic_loss(c(1, b1, b2), Z_with_intercept, y)))
loss_surface_l <- outer(gamma1_range, gamma2_range, 
                      Vectorize(function(b1, b2) logistic_loss_l(c(1, b1, b2), Zcor, y)))
loss_surface2 <- outer(gamma1_range, gamma2_range, 
                       Vectorize(function(b1, b2) pen_loss(c(1, b1, b2), Z_with_intercept, 
                                                           matrix(x), y)))

# Preparing data for ggplot
loss_df <- expand.grid(gamma1 = gamma1_range, gamma2 = gamma2_range)
loss_df$loss <- as.vector(loss_surface)
loss_df2 <- loss_df
loss_df2$loss <- as.vector(loss_surface2)
loss_dfl <- loss_df
loss_dfl$loss <- as.vector(loss_surface_l)

# Calculate correlation between y_hat and x_with_noise for each iteration
calculate_correlation <- function(gamma, Z, y, x) {
  y_hat <- plogis(Z %*% gamma)
  abs(cor(y_hat, x, use = "complete.obs"))
}

correlations_standard <- sapply(path_standard_newton, function(gammat) 
  calculate_correlation(gammat, Z_with_intercept, y, x))
correlations_custom <- sapply(path_custom_newton, function(gammat) 
  calculate_correlation(gammat, Z_with_intercept, y, x))
correlations_custom_old <- sapply(path_customold_newton, function(gammat) 
  calculate_correlation(gammat, Zcor, y, x))

# Prepare data for ggplot
path_standard_newton_df <- as.data.frame(do.call("rbind", path_standard_newton))
path_custom_newton_df <- as.data.frame(do.call("rbind", path_custom_newton))
path_customold_newton_df <- as.data.frame(do.call("rbind", path_customold_newton))

correlations_standard_df <- data.frame(iteration = 1:length(correlations_standard), 
                                       correlation = correlations_standard)
correlations_standard_df[is.na(correlations_standard_df$correlation),2] <- 0
correlations_custom_df <- data.frame(iteration = 1:length(correlations_custom), 
                                     correlation = correlations_custom)
correlations_custom_df[is.na(correlations_custom_df$correlation),2] <- 0
correlations_customold_df <- data.frame(iteration = 1:length(correlations_custom_old), 
                                        correlation = correlations_custom_old)
correlations_customold_df[is.na(correlations_customold_df$correlation),2] <- 0

# Plotting
p1 <- ggplot(loss_df, aes(x = gamma1, y = gamma2)) +
  geom_raster(aes(fill = loss)) +
  geom_contour(aes(z = loss), color = "white", alpha = 0.4) + 
  geom_path(data = path_standard_newton_df, mapping = aes(x = Zz1, y = Zz2), 
            color = "red", linetype = 2, alpha = 0.8) +
  geom_point(data = path_standard_newton_df, mapping = aes(x = Zz1, y = Zz2), color = "red",
             size = 0.4) +
  geom_point(data = data.frame(x = true_gammas[2], y = true_gammas[3]), 
             mapping = aes(x=x, y=y), shape = 4, size = 3, color = "black") +
  scale_fill_viridis_c() +
  labs(x = expression(gamma[1]), y = expression(gamma[2]), 
       title = "Without Orthogonalization") + theme_minimal() + 
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5),
        text = element_text(size = 12)
        # legend.key.height = unit(0.5, "cm"),  # Adjust height
        # legend.key.width = unit(2, "cm"),    # Adjust width
        # legend.text = element_text(size = rel(0.8)),  # Adjust text size
        # legend.title = element_text(size = rel(0.8))
        )

p2 <- ggplot(correlations_standard_df, aes(x = iteration, y = correlation)) +
  geom_line(color = "black") +
  geom_point(color = "red", size = 0.4) +
  labs(x = "Iteration", y = "Correlation of prediction\n and protected feature") + 
  theme_minimal() + ylim(0,0.3)

p3 <- ggplot(loss_dfl, aes(x = gamma1, y = gamma2)) +
  geom_raster(aes(fill = loss)) +
  geom_contour(aes(z = loss), color = "white", alpha = 0.4) + 
  geom_path(data = path_customold_newton_df, mapping = aes(x = Zz1, y = Zz2), 
            color = "red", linetype = 2, alpha = 0.8) +
  geom_point(data = path_customold_newton_df, mapping = aes(x = Zz1, y = Zz2), color = "red",
             size = 0.4) +
  geom_point(data = data.frame(x = path_customold_newton_final[[length(path_customold_newton_final)]][2], 
                               y = path_customold_newton_final[[length(path_customold_newton_final)]][3]), 
             mapping = aes(x=x, y=y), shape = 4, size = 3, color = "black") +
  scale_fill_viridis_c() +
  labs(x = expression(gamma[1]), y = expression(gamma[2]), 
       title = "Classical Orthogonalization") + theme_minimal() + 
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5),
        text = element_text(size = 12)
        # legend.key.height = unit(0.5, "cm"),  # Adjust height
        # legend.key.width = unit(2, "cm"),    # Adjust width
        # legend.text = element_text(size = rel(0.8)),  # Adjust text size
        # legend.title = element_text(size = rel(0.8))
        )

p4 <- ggplot(correlations_customold_df, aes(x = iteration, y = correlation)) +
  geom_line(color = "black") +
  geom_point(color = "red", size = 0.4) +
  labs(x = "Iteration", y = "") + 
  theme_minimal() + ylim(0,0.3)

p5 <- ggplot(loss_df2, aes(x = gamma1, y = gamma2)) +
  geom_raster(aes(fill = loss)) +
  geom_contour(aes(z = loss), color = "white", alpha = 0.4) + 
  geom_path(data = path_custom_newton_df, mapping = aes(x = Zz1, y = Zz2), 
            color = "red", linetype = 2, alpha = 0.8) +
  geom_point(data = path_custom_newton_df, mapping = aes(x = Zz1, y = Zz2), color = "red",
             size = 0.4) +
  geom_point(data = data.frame(x = path_custom_newton_final[[length(path_custom_newton_final)]][2], 
                               y = path_custom_newton_final[[length(path_custom_newton_final)]][3]), 
             mapping = aes(x=x, y=y), shape = 4, size = 3, color = "black") +
  scale_fill_viridis_c() +
  labs(x = expression(gamma[1]), y = expression(gamma[2]), 
       title = "Generalized Orthogonalization") + theme_minimal() + 
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5),
        text = element_text(size = 12)
        # legend.key.height = unit(0.5, "cm"),  # Adjust height
        # legend.key.width = unit(2, "cm"),    # Adjust width
        # legend.text = element_text(size = rel(0.8)),  # Adjust text size
        # legend.title = element_text(size = rel(0.8))
        )

p6 <- ggplot(correlations_custom_df, aes(x = iteration, y = correlation)) +
  geom_line(color = "black") +
  geom_point(color = "red", size = 0.4) +
  labs(x = "Iteration", y = "") + 
  theme_minimal() + ylim(0,0.3)

ppall <- grid.arrange(p1, p3, p5, p2, p4, p6, ncol = 3, nrow = 2, heights = c(1.5, 1))
                                                
ggsave(ppall, file = "demo_plot.pdf", width = 12, height = 4.5)
