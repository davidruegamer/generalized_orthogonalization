# 0. Libraries and functions

library(parallel)
set.seed(42)

# 1. Load the MNIST dataset

# mnist <- dataset_mnist()
# mnist$train$x <- mnist$train$x[mnist$train$y %in% c(0,9),,]
# mnist$test$x <- mnist$test$x[mnist$test$y %in% c(0,9),,]
# mnist$train$y <- mnist$train$y[mnist$train$y %in% c(0,9)]
# mnist$test$y <- mnist$test$y[mnist$test$y %in% c(0,9)]
# saveRDS(mnist, file = "MNIST.RDS")
mnist <- readRDS("MNIST.RDS")
train_images <- mnist$train$x
train_labels <- mnist$train$y
test_images <- mnist$test$x
test_labels <- mnist$test$y
train_red <- 0*mnist$train$y
test_red <- 0*mnist$test$y
test_red_actual <- 1*(test_labels==0)
rm(mnist); gc()

# 2. Colorize the MNIST data

colorize <- function(images, labels, test = FALSE) {
  colored_images <- array(0, dim = c(dim(images)[1], 28, 28, 3))
  
  for (i in 1:dim(images)[1]) {
    if (labels[i] == 0 & test==FALSE) {
      # Red channel for 0s
      train_red[i] <- 1
      colored_images[i, , , 1] <- images[i, , ]
      colored_images[i, , , 2] <- 0
      colored_images[i, , , 3] <- 0
    } else {
      # Either green or blue channel for others
      channel <- sample(c(2, 3), 1)
      colored_images[i, , , channel] <- images[i, , ]
    }
  }
  
  return(colored_images / 255) # Normalize
}

reps <- 10
max_epochs <- 100
batch_size <- 256
val_split <- 0.2
pat <- 5

res <- mclapply(1:reps, function(set){
  
  set.seed(set)
  
  library(keras)
  library(tensorflow)
  tensorflow::set_random_seed(set)
  
  train_images_colored <- colorize(train_images, train_labels) #, this_fac)
  test_images_colored <- colorize(test_images, test_labels, test = TRUE) #, this_fac)
  
  # 3. Creating a binary outcome (0 vs all)
  
  train_labels_binary <- as.integer(train_labels == 0)
  test_labels_binary <- as.integer(test_labels == 0)
  
  # 4. Constructing a CNN model including orthogonalization
  
  oz <- reticulate::import_from_path("orthog")
  
  inp <- layer_input(shape = c(28, 28, 3))
  first_layer <- inp %>% 
    layer_conv_2d(filters = 16, kernel_size = c(3,3), activation = 'relu')
  redinfo <- layer_input(shape = c(2))
  orth_first_layer <- oz$orthog_tf(first_layer, redinfo)
  outp <- orth_first_layer %>%
    layer_conv_2d(filters = 16, kernel_size = c(3,3), activation = 'relu') %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_flatten() %>%
    layer_dense(units = 64, activation = 'relu') %>%
    layer_dense(units = 1, activation = 'sigmoid') # Binary classification
  
  model <- keras_model(list(inp, redinfo), outp)
  
  model %>% compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = c('accuracy')
  )
  
  model_wo_correction <- keras_model_sequential() %>% 
    layer_conv_2d(filters = 16, kernel_size = c(3,3), activation = 'relu',
                  input_shape = c(28, 28, 3)) %>% 
    layer_conv_2d(filters = 16, kernel_size = c(3,3), activation = 'relu') %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_flatten() %>%
    layer_dense(units = 64, activation = 'relu') %>%
    layer_dense(units = 1, activation = 'sigmoid') # Binary classification
  
  model_wo_correction %>% compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = c('accuracy')
  )
  
  # 5. Training the model
  
  hist <- model %>% fit(
    list(train_images_colored, cbind(1,train_red)),
    train_labels_binary,
    epochs = max_epochs,
    batch_size = batch_size,
    validation_split = val_split,
    callbacks = list(
      callback_early_stopping(monitor = "val_accuracy",
                              patience = pat, 
                              restore_best_weights = TRUE)
    ),
    view_metrics = FALSE, verbose = FALSE
  )
  
  pred <- model %>% predict(list(test_images_colored, cbind(1, test_red)))
  testperf <- Metrics::accuracy(round(pred[,1]), test_labels_binary)
  
  hist2 <- model_wo_correction %>% fit(
    train_images_colored,
    train_labels_binary,
    epochs = max_epochs,
    batch_size = batch_size,
    validation_split = val_split,
    callbacks = list(
      callback_early_stopping(monitor = "val_accuracy",
                              patience = pat, 
                              restore_best_weights = TRUE)
    ),
    view_metrics = FALSE, verbose = FALSE
  )
  
  pred2 <- model_wo_correction %>% predict(test_images_colored)
  testperf2 <- Metrics::accuracy(round(pred2[,1]), test_labels_binary)
  
  # 6. Check significances
  
  conv_output <- keras_model(model$inputs[[1]], model$layers[[5]]$output)
  yhat <- conv_output$predict(test_images_colored)
  
  conv_output2 <- keras_model(model_wo_correction$inputs[[1]], model_wo_correction$layers[[1]]$output)
  yhat2 <- conv_output2$predict(test_images_colored)
  
  coefMat <- tf$matmul(tf$matmul(cbind(1, test_red_actual), 
                                 cbind(1, test_red_actual), transpose_a = TRUE),
                       cbind(1, test_red_actual), transpose_b = TRUE)
  
  beta <- as.array(tf$tensordot(coefMat, oz$orthog_tf(yhat, cbind(1,test_red_actual)), 
                                axes = list(c(1L), c(0L))))[2,,,]
  beta2 <- as.array(tf$tensordot(coefMat, yhat2, axes = list(c(1L), c(0L))))[2,,,]
  
  # Return
  
  list(acc = 
         data.frame(value = c(tail(hist$metrics$accuracy,1), 
                              tail(hist$metrics$val_accuracy,1),
                              testperf,
                              tail(hist2$metrics$accuracy,1), 
                              tail(hist2$metrics$val_accuracy,1),
                              testperf2),
                    type = rep(c("w/ correction", "w/o correction"), each = 3),
                    what = rep(c("train", "valid", "test"), 2),
                    rep = set),
       coef = cbind(as.data.frame(rbind(summary(c(beta)),
                                        summary(c(beta2)))),
                    type = c("w/ correction", "w/o correction"))
         )
  
}, mc.cores = 4)

saveRDS(res, file = "result_MNIST.RDS")

ress <- do.call("rbind", lapply(res, "[[", 1))
coefs <- do.call("rbind", lapply(res, "[[", 2))

library(ggplot2)
library(dplyr)

my_palette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", 
                "#0072B2", "#D55E00", "#CC79A7")

(g1 <- ggplot(ress %>% mutate(
  what = factor(what, levels=c("train", "valid", "test"),
                labels = c("train", "validation", "test")),
  type = factor(type, levels=c("w/o correction", "w/ correction"))
  ), 
  aes(y = value, x = what, colour = type)) + 
  geom_boxplot() + theme_bw() + 
  scale_colour_manual(values = my_palette) + 
  theme(
    legend.title = element_blank(),
    text = element_text(size = 13),
    axis.text.x=element_text(colour="black")
    # legend.position="bottom"
  ) + xlab("") + ylab("Accuracy"))

ggsave("result_mnist.pdf", width = 5, height = 3)

(g2 <- ggplot(aggregate(coefs %>% 
                   select(-type),
                 list(type = coefs$type), mean) %>% 
         mutate(type = factor(type, levels=c("w/o correction", "w/ correction"),
                              labels=c("w/o corr.", "w/ corr."))), 
       aes(x = as.factor(type), colour = type)) +
  geom_boxplot(aes(
    lower = `1st Qu.`, 
    upper = `3rd Qu.`, 
    middle = Median, 
    ymin = `Min.`, 
    ymax = `Max.`),
    stat = "identity") + theme_bw() + 
  scale_colour_manual(values = my_palette) + 
  theme(
    legend.title = element_blank(),
    text = element_text(size = 13),
    axis.text.x=element_text(colour="black"),
    legend.position = "none"
  ) + xlab("") + ylab("Influence of color\n on hidden layer"))

ggsave("result_mnist_coef.pdf", width = 4, height = 3)

library(gridExtra)

p1 <- grid.arrange(g2, g1, widths = c(1, 3))

ggsave(p1, file="result_mnist_both.pdf", width = 9, height = 2.5)
