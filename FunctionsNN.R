# Initialization
#####################################################
# p - dimension of input layer
# hidden_p - dimension of hidden layer
# K - number of classes, dimension of output layer
# scale - magnitude for initialization of W_k (standard deviation of normal)
# seed - specified seed to use before random normal draws
initialize_bw <- function(p, hidden_p, K, scale = 1e-3, seed = 12345){
  set.seed(seed)
  # [ToDo] Initialize intercepts as zeros
  b1 <- rep(0, hidden_p)
  b2 <- rep(0, K)
  # [ToDo] Initialize weights by drawing them iid from Normal
  # with mean zero and scale as sd
  W1 <- matrix(rnorm(p * hidden_p, mean = 0, sd = scale), nrow = p, ncol = hidden_p)
  W2 <- matrix(rnorm(hidden_p * K, mean = 0, sd = scale), nrow = hidden_p, ncol = K)
  # Return
  return(list(b1 = b1, b2 = b2, W1 = W1, W2 = W2))
}

# Function to calculate loss, error, and gradient strictly based on scores
# with lambda = 0
#############################################################
# scores - a matrix of size n by K of scores (output layer)
# y - a vector of size n of class labels, from 0 to K-1
# K - number of classes
loss_grad_scores <- function(y, scores, K){
  n <- nrow(scores)
  #getting Y in a more usable format, by encoding 1 when it is of the given class
  y_encode <- matrix(0, nrow = n, ncol = K)
  y_encode[cbind(1:n, y + 1)] <- 1
  # [ToDo] Calculate loss when lambda = 0
  exp_scores <- exp(scores)
  probabilities <- exp_scores / rowSums(exp_scores)
  
  loss = -mean(rowSums(y_encode * log(probabilities)))
  
  # [ToDo] Calculate misclassification error rate (%)
  # when predicting class labels using scores versus true y
  predictions = (apply(probabilities, 1, which.max) - 1) 
  error = mean(predictions != y) * 100
  
  # [ToDo] Calculate gradient of loss with respect to scores (output)
  # when lambda = 0
  grad = (probabilities - y_encode) / n 
  
  # Return loss, gradient and misclassification error on training (in %)
  return(list(loss = loss, grad = grad, error = error))
}

# One pass function
################################################
# X - a matrix of size n by p (input)
# y - a vector of size n of class labels, from 0 to K-1
# W1 - a p by h matrix of weights
# b1 - a vector of size h of intercepts
# W2 - a h by K matrix of weights
# b2 - a vector of size K of intercepts
# lambda - a non-negative scalar, ridge parameter for gradient calculations
one_pass <- function(X, y, K, W1, b1, W2, b2, lambda){
  n <- nrow(X)
  # [To Do] Forward pass
  # From input to hidden 
  H1 <- X %*% W1 + matrix(b1, nrow=n, ncol=length(b1), byrow=TRUE)
  # ReLU
  H <- (H1 + abs(H1))/2
  # From hidden to output scores
  scores <- H %*% W2 + matrix(b2, nrow=n, ncol=length(b2), byrow=TRUE)
  
  # [ToDo] Backward pass
  # Get loss, error, gradient at current scores using loss_grad_scores function
  out <- loss_grad_scores(y, scores, K)
  grad_scores <- out$grad
  # Get gradient for 2nd layer W2, b2 (use lambda as needed)
  dW2 <- crossprod(H, grad_scores) + lambda * W2
  db2 <- colSums(grad_scores)
  # Get gradient for hidden, and 1st layer W1, b1 (use lambda as needed)
  dhidden <- tcrossprod(grad_scores, W2) * (H1 > 0)
  dW1 <- crossprod(X, dhidden) + lambda * W1
  db1 <- colSums(dhidden)
  # Return output (loss and error from forward pass,
  # list of gradients from backward pass)
  return(list(loss = out$loss, error = out$error, grads = list(dW1 = dW1, db1 = db1, dW2 = dW2, db2 = db2)))
}

# Function to evaluate validation set error
####################################################
# Xval - a matrix of size nval by p (input)
# yval - a vector of size nval of class labels, from 0 to K-1
# W1 - a p by h matrix of weights
# b1 - a vector of size h of intercepts
# W2 - a h by K matrix of weights
# b2 - a vector of size K of intercepts
evaluate_error <- function(Xval, yval, W1, b1, W2, b2){
  # [ToDo] Forward pass to get scores on validation data
  forward_output <- one_pass(Xval, yval, length(b2),W1, b1, W2, b2, 0)
  # [ToDo] Evaluate error rate (in %) when 
  # comparing scores-based predictions with true yval
  error <- forward_output$error
  
  return(error)
}


# Full training
################################################
# X - n by p training data
# y - a vector of size n of class labels, from 0 to K-1
# Xval - nval by p validation data
# yval - a vector of size nval of of class labels, from 0 to K-1, for validation data
# lambda - a non-negative scalar corresponding to ridge parameter
# rate - learning rate for gradient descent
# mbatch - size of the batch for SGD
# nEpoch - total number of epochs for training
# hidden_p - size of hidden layer
# scale - a scalar for weights initialization
# seed - for reproducibility of SGD and initialization
NN_train <- function(X, y, Xval, yval, lambda = 0.01,
                     rate = 0.01, mbatch = 20, nEpoch = 100,
                     hidden_p = 20, scale = 1e-3, seed = 12345){
  # Get sample size and total number of batches
  n = length(y)
  nBatch = floor(n/mbatch)

  # [ToDo] Initialize b1, b2, W1, W2 using initialize_bw with seed as seed,
  # and determine any necessary inputs from supplied ones
  initialize_parameters <- initialize_bw(ncol(X), hidden_p, length(unique(y)), scale = scale, seed = seed)
  b1 <- initialize_parameters$b1
  b2 <- initialize_parameters$b2
  W1 <- initialize_parameters$W1
  W2 <- initialize_parameters$W2
  # Initialize storage for error to monitor convergence
  error = rep(NA, nEpoch)
  error_val = rep(NA, nEpoch)
  
  # Set seed for reproducibility
  set.seed(seed)
  # Start iterations
  for (epoch in 1:nEpoch){
    # Allocate bathes
    batchids = sample(rep(1:mbatch, length.out = n), size = n)
    batch_errors <- numeric(mbatch)
    # [ToDo] For each batch
    #  - do one_pass to determine current error and gradients
    #  - perform SGD step to update the weights and intercepts
    for (batch in 1:mbatch){
      batch_indices <- which(batchids == batch)
      X_batch <- X[batch_indices, , drop=FALSE]
      y_batch <- y[batch_indices]
      forward_out <- one_pass(X_batch, y_batch, length(b2), W1, b1, W2, b2, lambda)
      grads <- forward_out$grads
      batch_errors[batch] <- forward_out$error
      W1 <- W1 - rate * grads$dW1
      W2 <- W2 - rate * grads$dW2
      b1 <- b1 - rate * grads$db1
      b2 <- b2 - rate * grads$db2
      
    }
    # [ToDo] In the end of epoch, evaluate
    # - average training error across batches
    # - validation error using evaluate_error function
    error[epoch] <- mean(batch_errors)
    error_val[epoch] <- evaluate_error(Xval, yval, W1, b1, W2, b2)
  }
  # Return end result
  return(list(error = error, error_val = error_val, params =  list(W1 = W1, b1 = b1, W2 = W2, b2 = b2)))
}