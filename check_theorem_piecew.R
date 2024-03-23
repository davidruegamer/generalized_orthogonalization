# Define the ReLU function
h <- function(x) {
  return(pmax(0, x))
}

# Generate or specify matrices X and Z
# Example: Let's use random matrices for illustration
set.seed(123)  # For reproducibility
X <- matrix(rnorm(25), nrow=5)  # 5x5 matrix
Z <- matrix(rnorm(25), nrow=5)  # 5x5 matrix

# Calculate the projection matrix P_X and its orthogonal complement P_X_bot
P_X <- X %*% solve(t(X) %*% X) %*% t(X)
P_X_bot <- diag(nrow(X)) - P_X

# Define vectors gamma and beta
# Example: Random vectors
gamma <- rnorm(nrow(Z))
beta <- rnorm(ncol(X))

# Calculate Z^c
Z_c <- P_X_bot %*% Z

# Evaluate each term in the equation
term1 <- h(Z_c %*% gamma) %*% h(X %*% beta)
term2 <- h(-Z_c %*% gamma) %*% h(X %*% beta)
term3 <- h(Z_c %*% gamma) %*% h(-X %*% beta)
term4 <- h(-Z_c %*% gamma) %*% h(-X %*% beta)

# Check if the sum of the terms equals zero
sum_terms <- term1 + term2 + term3 + term4
result <- c(sum_terms < 1e-10)

# Print the result
print(result)
