n_train <- 200  # Number of training samples
n_val <- 50     # Number of validation samples

# Class 0
X0_train <- matrix(rnorm(n_train, mean = 2), ncol = 2)
y0_train <- rep(0, n_train / 2)

X0_val <- matrix(rnorm(n_val, mean = 2), ncol = 2)
y0_val <- rep(0, n_val / 2)

# Class 1
X1_train <- matrix(rnorm(n_train, mean = -2), ncol = 2)
y1_train <- rep(1, n_train / 2)

X1_val <- matrix(rnorm(n_val, mean = -2), ncol = 2)
y1_val <- rep(1, n_val / 2)

# Combine training data
X_train <- rbind(X0_train, X1_train)
y_train <- c(y0_train, y1_train)

# Combine validation data
X_val <- rbind(X0_val, X1_val)
y_val <- c(y0_val, y1_val)

# Train
out2 <- NN_train(X = X_train, y = y_train, Xval = X_val, yval = y_val,
                   lambda = 0.01, rate = 0.1, mbatch = 20, nEpoch = 50,
                   hidden_p = 10, scale = 1e-2, seed = 12345)
plot(1:length(out2$error), out2$error, ylim = c(0, 70))
lines(1:length(out2$error_val), out2$error_val, col = "red")

# Evaluate error on testing data
test_error = evaluate_error(X_train, y_train, out2$params$W1, out2$params$b1, out2$params$W2, out2$params$b2)
test_error #should be 0 