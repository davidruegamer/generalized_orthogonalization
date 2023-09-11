source("../../simulations/common/functions.R")
library(dplyr)

### data prep
data <- read.table("adult.data", sep = ",")
data <- as.data.frame(sapply(data, trimws))
colnames(data) <- c("age", "workclass", "fnlwgt", "education",
                    "educaiton_num", "marital_status", "occupation", 
                    "relationship", "race", "sex", "capital_gain",
                    "capital_loss", "hours_per_week", "native_country",
                    "income")
data[,c(1,3,5,11:13)] <- lapply(data[,c(1,3,5,11:13)], as.numeric)
data[,-c(1,3,5,11:13)] <- lapply(data[,-c(1,3,5,11:13)], as.factor)
data$income <- as.numeric(data$income)-1

data <- data %>% filter(native_country == "United-States")

### define and fit prediction model
prediction_model <- glm(income ~ age + workclass + education + 
                          marital_status + relationship + 
                          hours_per_week, data = data, 
                        family = binomial)

### get predicted values as scores and probabilities
yhat_score <- predict(prediction_model, type = "link")
yhat_prob <- predict(prediction_model, type = "response")

### define and fit evaluation models
linear_proj_model <- lm(yhat ~ sex + race, 
                        data = cbind(data, yhat = yhat_score))
anova(linear_proj_model) # significant influence of sex and race
nonlinear_proj_model <- glm(yhat ~ sex + race, 
                            data = cbind(data, yhat = yhat_prob),
                            family = binomial)
summary(nonlinear_proj_model)$coefficients[,c(1,4)] # significant influence of sex and race

### extract design matrices
X <- model.matrix(linear_proj_model)
Z <- model.matrix(prediction_model)

### correct model
yhatc_score <- orthog(to_orthog = yhat_score, orthog_with = X)
Zc_prob <- correct_Z(X, Z, data$income, binomial, printevery = 1, 
                     maxdiff = 1e-10)
yhatc_prob <- glm(data$income ~ -1 + Zc_prob,
                  family = binomial)

### check again using evaluation model
linear_proj_model <- lm(yhat ~ sex + race, 
                        data = cbind(data, yhat = yhatc_score))
summary(linear_proj_model)$coefficients[,c(1,4)] # betas close to zero
anova(linear_proj_model) # no influence significant
nonlinear_proj_model <- glm(yhat ~ sex + race, 
                            data = cbind(data, yhat = predict(yhatc_prob, type="response")),
                            family = binomial)
summary(nonlinear_proj_model)$coefficients[,c(1,4)] # no influence significant and betas close to zero


