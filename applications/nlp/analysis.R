library(tidyverse)
library(lubridate)
library(textTinyR)
library(tidytext)
library(tm)
library(text2vec)
library(data.table)
library(caret)
library(keras)
library(tensorflow)
library(parallel)

set.seed(42)

### Data Pre-Processing
nr_words <- 1e4
embedding_size <- 1e2
maxlen <- 1e2

if(!file.exists("movies_processed.RDS") |
   !file.exists("words.RDS") | 
   !file.exists("mov_ov.RDS")){
  
  gender <- readRDS("sex.RDS")
  movies <- read_csv("movies_metadata.csv")
  
  # merge gender and data
  movie_data <- left_join(gender, movies, by="id", multiple="first")
  rm(gender,movies); gc()
  
  movie_data <- movie_data %>% select(
    gender_main_actor,
    gender_director,
    overview,
    release_date,
    vote_average,
    vote_count
  ) %>% 
    mutate(
      release_date = as.numeric(as.Date("2020-01-01") - as.Date(release_date))
    )
  
  tokenizer <- text_tokenizer(num_words = nr_words)
  
  # remove stopwords
  data("stop_words")
  stopwords_regex <- paste(c(stopwords('en'), stop_words$word),
                           collapse = '\\b|\\b')
  stopwords_regex <- paste0('\\b', stopwords_regex, '\\b')
  movie_data$overview <- tolower(movie_data$overview)
  movie_data$overview <-
    stringr::str_replace_all(movie_data$overview, stopwords_regex, '')
  movie_data$overview <- gsub('[[:punct:] ]+', ' ', movie_data$overview)
  
  saveRDS(movie_data$overview, file = "mov_ov.RDS")
  
  tokenizer %>% fit_text_tokenizer(movie_data$overview)
  
  # text to sequence
  text_seqs <- texts_to_sequences(tokenizer, movie_data$overview)
  
  # pad text sequences
  text_padded <- text_seqs %>%
    pad_sequences(maxlen = maxlen, truncating = "post")
  
  # save words for later
  words <- tibble(word = names(tokenizer$word_index),
                  id = as.integer(unlist(tokenizer$word_index)))
  
  words <- words %>%
    filter(id <= tokenizer$num_words) %>%
    arrange(id)
  
  saveRDS(words, file = "words.RDS")
  rm(words)
  gc()
  
  # text sequences as list of one array
  text_embd <-
    list(texts = array(text_padded, dim = c(NROW(movie_data), maxlen)))
  
  # create input list
  data <- append(movie_data, text_embd)
  rm(movie_data); gc()
  
  saveRDS(data, file="movies_processed.RDS")
  
}else{
  
  ### Load after pre-processing
  
  tokenizer <- text_tokenizer(num_words = nr_words)
  mov <- readRDS("mov_ov.RDS")
  tokenizer |> fit_text_tokenizer(mov)
  words <- readRDS("words.RDS")
  data <- readRDS("movies_processed.RDS")

}

### Data splitting

complete_indices <- which(complete.cases(
  as.data.frame(data[c(1:2,6)])))

data <- lapply(data, function(feat) if(is.null(dim(feat))) 
  feat[complete_indices] else feat[complete_indices,])

y <- matrix(data$vote_count)

# Split the data into training and test sets
set.seed(123)
trainIndex <- createDataPartition(y, p = .8, 
                                  list = FALSE, 
                                  times = 1)

Z_train <- data$texts[trainIndex, ]
Z_test <- data$texts[-trainIndex, ]

X_train <- as.data.frame(data[c("gender_main_actor", "gender_director")])[trainIndex, ]
X_test <- as.data.frame(data[c("gender_main_actor", "gender_director")])[-trainIndex, ]
X_train <- as.matrix(X_train) - 1
X_test <- as.matrix(X_test) - 1

y_train <- y[trainIndex,]
y_test <- y[-trainIndex,]


### Build the Keras model
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = nr_words, output_dim = embedding_size) |>
  layer_lstm(units = 50, return_sequences = FALSE, activation = "relu") |>
  layer_dropout(rate = 0.1) |>
  layer_dense(25, activation = "relu") |>
  layer_dropout(rate = 0.2) |>
  layer_dense(5, activation = "relu") |>
  layer_dropout(rate = 0.3) |>
  layer_dense(1, activation = 'exponential')

# Compile the model
model %>% compile(
  optimizer = optimizer_adam(learning_rate = 1e-6),
  loss = 'poisson', 
  metrics = c('mean_squared_error')
)

hist <- model %>% fit(x = Z_train, y = y_train,
                      epochs = 1000,
                      batch_size = 128,
                      validation_split = 0.2,
                      callbacks = list(
                        callback_early_stopping(patience = 10, 
                                                restore_best_weights = TRUE)
                      ),
                      view_metrics = FALSE)

fits <- model %>% predict(Z_train)
preds <- model %>% predict(Z_test)

saveRDS(list(fits=fits,preds=preds), file="fits_preds_wo_corr.RDS")

summary(glm(fits ~ X_train, family = poisson()))
summary(glm(preds ~ X_test, family = poisson()))

Metrics::rmse(preds, y_test)

### With correction

oz <- reticulate::import_from_path("orthog")

inp <- layer_input(shape = c(100))
first_layer <- inp %>% 
  layer_embedding(input_dim = nr_words, output_dim = embedding_size)
redinfo <- layer_input(shape = c(3))
orth_first_layer <- oz$orthog_tf(first_layer, redinfo)
outp <- orth_first_layer %>%
  layer_lstm(units = 50, return_sequences = FALSE, activation = "relu") |>
  layer_dropout(rate = 0.1) |>
  layer_dense(25, activation = "relu") |>
  layer_dropout(rate = 0.2) |>
  layer_dense(5, activation = "relu") |>
  layer_dropout(rate = 0.3) |>
  layer_dense(1, activation = 'exponential')

model <- keras_model(list(inp, redinfo), outp)

model %>% compile(
  optimizer = optimizer_adam(learning_rate = 1e-6),
  loss = 'poisson', 
  metrics = c('mean_squared_error')
)

hist_corr <- model %>% fit(x = list(Z_train, cbind(1,X_train)), y = y_train,
                           epochs = 1000,
                           batch_size = 128,
                           validation_split = 0.2,
                           callbacks = list(
                             callback_early_stopping(patience = 10, restore_best_weights = TRUE),
                             callback_terminate_on_naan()
                           ),
                           view_metrics = FALSE)

fits_corr <- model %>% predict(list(Z_train, cbind(1, X_train)))
preds_corr <- model %>% predict(list(Z_test, cbind(1, X_test)))

saveRDS(list(fits=fits_corr,preds=preds_corr), file="fits_preds_w_corr.RDS")

summary(glm(fits_corr ~ X_train, family = poisson()))
summary(glm(preds_corr ~ X_test, family = poisson()))

Metrics::rmse(preds_corr, y_test)/Metrics::rmse(preds, y_test)
