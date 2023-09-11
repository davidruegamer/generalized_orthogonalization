rm(list = ls())

set.seed(223)
nr_sims <- 16
sd_sample = 1
p = 100
n = 100
q = 2 # cant change this

### orthog proj
orthog <- function(to_orthog, orthog_with, Q = NULL)
{
  ### construct projection matrices
  if(is.null(Q))
    Q <- qr.Q(qr(orthog_with))
  
  return(to_orthog - Q%*%(crossprod(Q, to_orthog)))
  
}

### sampling fun
sample_fun = function(n, m) rnorm(n, m, sd = 1)

### activation fun
h = function(x) return(pmax(0, x))

### loss fun
loss_function <- function(beta, x, y) {
  predictions = h(x %*% beta)
  mse = mean((y - predictions)^2)
  return(mse)
}


true_coef <- rnorm(p, sd = sd_sample) 
hidden_coef <- rnorm(q)
Z <- matrix(rnorm(n * p, sd = sd_sample), nrow = n)
X <- matrix(rnorm(n * q, sd = 0.1), nrow = n)
X[, 1:pmin(p,q)] <- X[, 1:pmin(p,q)] + 2 * Z[, 1:pmin(p,q)]
Zc <- orthog(Z, cbind(1,X))
true_add_pred <- Z%*%true_coef + X%*%hidden_coef

par(mfrow=c(4,4))

for(i in 1:nr_sims){
  
  y <- sample_fun(n, h(true_add_pred))
  
  result = optim(rnorm(length(true_coef)), loss_function, x = Zc, y = y, method = "BFGS")
  yhatc <- Zc%*%result$par
  
  ### loss surface
  beta1_vals <- seq(-1.5, 1.5, by = 0.02)
  beta2_vals <- seq(-1.5, 1.5, by = 0.02)
  grid_length <- length(beta1_vals)
  loss_values <- matrix(0, nrow = grid_length, ncol = grid_length)
  
  # Populate the loss_values matrix
  for (i in 1:grid_length) {
    for (j in 1:grid_length) {
      beta <- c(beta1_vals[i], beta2_vals[j])
      loss_values[i, j] <- loss_function(beta, X, yhatc)
    }
  }
  
  # Find the coordinates of the minimum loss
  min_loss_index <- which(loss_values == min(loss_values), arr.ind = TRUE)
  min_beta1 <- beta1_vals[min_loss_index[1, 1]]
  min_beta2 <- beta2_vals[min_loss_index[1, 2]]
  
  # Create the surface plot
  contour(beta1_vals, beta2_vals, loss_values, xlab = "Beta 1", ylab = "Beta 2", main = "Contour plot of Loss")
  
  # Add a red point for the minimum loss
  points(min_beta1, min_beta2, col = "red", pch = 19)
  
  # Add vertical and horizontal lines connecting the minimum to the axes
  abline(h = min_beta2, col = "red", lty = 2)
  abline(v = min_beta1, col = "red", lty = 2)
  
  # Add text to display the minimum loss value next to the red point
  text(min_beta1, min_beta2, labels = sprintf("Min Loss: %.4f", min(loss_values)), col = "red", pos = 1)
  
  # Calculate and display the loss value at c(0,0)
  loss_at_zero <- loss_function(c(0,0), X, yhatc)
  text(0.25, -0.25, labels = sprintf("Loss at (0,0): %.4f", loss_at_zero), col = "red", pos = 1)
  
}