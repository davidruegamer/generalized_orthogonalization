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
yhat_score <- matrix(yhat_score)
colnames(yhat_score) <- "yhats"
yhatc_score <- orthog(to_orthog = yhat_score, orthog_with = X)
# yhatc_prob <- correct_Z(X, Z, data$income, "binomial", "pred", 
#                         conv_threshold = 5e-2, fac_improvem = 1000)
yhatc_prob <- lagrangianConstr(X, Z, data$income, "binomial", "pred")
  
### check again using evaluation model
linear_proj_model <- lm(yhat ~ sex + race, 
                        data = cbind(data, yhat = c(yhatc_score)))
summary(linear_proj_model)$coefficients[,c(1,4)] # betas close to zero
anova(linear_proj_model) # no influence significant
nonlinear_proj_model <- glm(yhat ~ sex + race, 
                            data = cbind(data, yhat = yhatc_prob),
                            family = binomial)
summary(nonlinear_proj_model)$coefficients[,c(1,4)] # no influence significant and betas close to zero

Metrics::accuracy(yhatc_prob, data$income)

### Xu et al 
w_x <- yhat_prob
w_1 <- predict(glm(income ~ sex + race,
                   data = data),
               type="response")
prob0 <- (data$income==0)/nrow(data)
prob1 <- (data$income==1)/nrow(data)

yhatc_prob_xu <- w_x/w_1 * prob1 / (w_x/w_1 * prob1 + (1-w_x)/(1-w_1) * prob0)
eval_model_xu <- glm(yhat ~ sex + race, 
                            data = cbind(data, yhat = yhatc_prob_xu),
                            family = binomial)
summary(eval_model_xu)$coefficients[,c(1,4)] 

Metrics::accuracy(yhatc_prob_xu, data$income)
