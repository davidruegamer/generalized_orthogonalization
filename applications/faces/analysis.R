library(tidyverse)
library(reticulate)
library(Metrics)
library(corpcor)
np <-import("numpy")
source("../../simulations/common/functions.R")

## Read meta data
meta <- read_csv("utk_meta.csv")
meta$sex <- as.factor(meta$sex)
meta$race <- as.factor(meta$race)
meta_train <- meta %>% filter(split=="train")
meta_test <- meta %>% filter(split=="test")
rm(meta); gc()



if(file.exists("emb_svd.RDS")){
  
  emb <- np$load("utk_emb_train.npy")
  emb_svd <- readRDS("emb_svd.RDS")
  red <- 32
  
}else{  
  
  emb <- np$load("utk_emb_train.npy")
  
  ## SVD
  emb_svd <- fast.svd(emb)
  saveRDS(emb_svd, file="emb_svd.RDS")
  plot(cumsum(emb_svd$d^2/sum(emb_svd$d^2)), type="b")
  (red <- min(which(cumsum(emb_svd$d^2/sum(emb_svd$d^2))>0.95)))
  
}

# create reduced embedding from SVD  
emb <- emb%*%emb_svd$v[,1:red]
emb_test <- tcrossprod(np$load("utk_emb_test.npy"), 
                       emb_svd$v[1:red,])

## Combine
emb <- cbind(as.data.frame(emb), meta_train)
emb_test <- cbind(as.data.frame(emb_test), meta_test)
rm(meta_train, meta_test); gc()

## Create response 
name_resp <- "sex"
emb$resp <- emb[,name_resp]
emb_test$resp <- emb_test[,name_resp]

## Fit model without protected features
formla <- paste0("resp ~ 1 + ",
                 paste(paste0("V", 1:red), collapse = " + "))

mod <- glm(as.formula(formla), family = "binomial",
           data = emb)

## Prediction performance
pred <- predict(mod, emb_test, type = "response")
(auc <- auc(emb_test$resp, c(1-pred)))

# check how much predictions can be explained
# by protected features
emb$pred_mod_probs <- predict(mod, type = "response")

mod_explained_protected_prob <- glm(
  pred_mod_probs ~ 1 + age + race,
  data = emb, family = "binomial"
)

# all significant
summary(mod_explained_protected_prob)

## Now adjust using C^h
# gamma <- correct_Z(model.matrix(~ ., data = emb[,c("age", "race")]), 
#                    as.matrix(emb[,paste0("V", 1:red)]), 2-as.numeric(emb$resp), 
#                    fam = "binomial", "gamma", fac_improvem = 100000, lr = 0.01,
#                    conv_threshold = 1e-3, 
#                    gamma_start = c(qlogis(mean(2-as.numeric(emb$resp))), rep(0,red-1)))
gamma <- lagrangianConstr(model.matrix(~ ., data = emb[,c("age", "race")]), 
                          as.matrix(emb[,paste0("V", 1:red)]), 2-as.numeric(emb$resp), 
                          fam = "binomial", "gamma")

# saveRDS(emdc, file="corrected_emb.RDS")
# 
# emb[,paste0("V", 1:red)] <- emdc
# 
# mod_fixed <- glm(as.formula(formla), family = "binomial",
#                  data = emb)

## Prediction performance
pred_fixed <- plogis(as.matrix(emb_test[,paste0("V", 1:red)])%*%gamma)
(auc_fixed <- auc(emb_test$resp, c(pred_fixed)))


# check how much predictions can be explained
# by protected features
# emb$pred_mod_fixed <- predict(mod_fixed, type = "response")

mod_explained_fixed <- glm(
  plogis(as.matrix(emb[,paste0("V", 1:red)])%*%gamma) ~ 1 + age + race,
  data = emb, family = "binomial"
)

# run inference
summary(mod_explained_fixed)
