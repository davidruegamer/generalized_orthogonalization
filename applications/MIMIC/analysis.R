# library(mgcv)
library(corpcor)
library(readr)
library(reticulate)
library(Metrics)
np <-import("numpy")
source("../../simulations/common/functions.R")

## Read meta data
meta <- read_csv("mimic_cfm_train_meta.csv")
meta_test <- read_csv("mimic_cfm_test_meta.csv")

## Label values
# 1.0 :  The label was positively mentioned in the associated study, and is 
#        present in one or more of the corresponding images, e.g. 
#        "A large pleural effusion"
# 0.0 :  The label was negatively mentioned in the associated study, and 
#        therefore should not be present in any of the corresponding images
#        e.g. "No pneumothorax."
# -1.0 : The label was either: 
#        (1) mentioned with uncertainty in the report, and therefore may or 
#            may not be present to some degree in the corresponding image, or 
#        (2) mentioned with ambiguous language in the report and it is unclear 
#            if the pathology exists or not
# Explicit uncertainty: "The cardiac size cannot be evaluated."
# Ambiguous language: "The cardiac contours are stable."
# Missing (empty element) - No mention of the label was made in the report

## ViewPosition: Filter for PA and AP -> already the case

meta$sex <- as.factor(meta$sex)
meta_test$sex <- as.factor(meta_test$sex)

## race: Merge Asian, White, Black
meta$race <- as.factor(meta$race)
meta_test$race <- as.factor(meta_test$race)

## Read embedding
# how many of the singular values should be used
# (based on SVD gives the following explained variances
#  for different amount of columns)
# red <- 19 # 95% variance
# red <- 73 # 97,5% variance
red <- 111 # 98% variance
# red <- 311 # 99% variance

if(file.exists("emb_svd.RDS")){
  
  emb <- np$load("mimic_cfm_train_emb.npy")
  emb_svd <- readRDS("emb_svd.RDS")
  
}else{  
  
  emb <- np$load("mimic_cfm_train_emb.npy")
  
  ## SVD
  emb_svd <- fast.svd(emb)
  saveRDS(emb_svd, file="emb_svd.RDS")
  plot(cumsum(emb_svd$d^2/sum(emb_svd$d^2)), type="b")
  # (red <- min(which(cumsum(emb_svd$d^2/sum(emb_svd$d^2))>0.98)))

}

# create reduced embedding from SVD  
emb <- emb%*%emb_svd$v[,1:red]
emb_test <- tcrossprod(np$load("mimic_cfm_test_emb.npy"), 
                       emb_svd$v[1:red,])

## Combine
emb <- cbind(as.data.frame(emb), meta)
emb_test <- cbind(as.data.frame(emb_test), meta_test)
rm(meta, meta_test); gc()

## Create response 
name_resp <- "Pleural Effusion"
emb$resp <- emb[,name_resp]
emb_test$resp <- emb_test[,name_resp]

## Fit model without protected features
formla <- paste0("resp ~ 1 + ",
                  paste(paste0("V", 1:red), collapse = " + "))

mod <- glm(as.formula(formla), family = "binomial",
            data = emb)

## Prediction performance
pred <- predict(mod, emb_test, type = "response")
(auc <- auc(emb_test$resp, c(pred)))

# check how much predictions can be explained
# by protected features
emb$pred_mod <- predict(mod)

mod_explained_protected <- glm(
  pred_mod ~ 1 + age + sex + race,
  data = emb
)

# all significant
summary(mod_explained_protected)

emb$pred_mod_probs <- predict(mod, type = "response")

mod_explained_protected_prob <- glm(
  pred_mod_probs ~ 1 + age + sex + race,
  data = emb, family = "binomial"
)

# all significant
summary(mod_explained_protected_prob)

## Now adjust embedding using C^l
feat_mat <- model.matrix(~ -1 + age + sex + race,
                         data = emb)
q_mat <- qr.Q(qr(feat_mat))

# replace embedding
rhs <- crossprod(q_mat, as.matrix(emb[,paste0("V", 1:red)]))
proj_emb <- q_mat%*%rhs
emb_org <- emb[,paste0("V", 1:red)]
emb[,paste0("V", 1:red)] <- emb[,paste0("V", 1:red)] - proj_emb  

## Now try again to see how much protected features are in the predictions
mod2_fixed <- glm(as.formula(formla), family = "binomial",
                  data = emb)

## Prediction performance
pred2_fixed <- predict(mod2_fixed, emb_test, type = "response")
(auc_fixed <- auc(emb_test$resp, c(pred2_fixed)))

# check how much predictions can be explained
# by protected features
emb$pred_mod2_fixed <- predict(mod2_fixed, type = "response")

mod2_explained_fixed <- glm(
  pred_mod2_fixed ~ 1 + age + sex + race,
  data = emb, family = "binomial"
)

# run inference
summary(mod2_explained_fixed)

## Now adjust using C^h
gamma_c <- correct_Z(model.matrix(~ ., data = emb[,c("age", "sex", "race")]), 
                     as.matrix(emb_org), emb$resp, 
                     fam = "binomial", what = "gamma")
pred3_fixed <- plogis(as.matrix(emb_test[,1:ncol(emb_org)])%*%gamma_c)
# 
# saveRDS(emdc, file="corrected_emb.RDS")
# 
# emb[,paste0("V", 1:red)] <- emdc
# 
# mod3_fixed <- glm(as.formula(formla), family = "binomial",
#                   data = emb)
# 
# ## Prediction performance
# pred3_fixed <- predict(mod3_fixed, emb_test, type = "response")
((auc3_fixed <- auc(emb_test$resp, c(pred3_fixed))) - auc_fixed)/auc_fixed
  

# check how much predictions can be explained
# by protected features
# emb$pred_mod3_fixed <- predict(mod3_fixed, type = "response")

mod3_explained_fixed <- glm(
  plogis(as.matrix(emb_org) %*% gamma_c) ~ 1 + age + sex + race,
  data = emb, family = "binomial"
)

# run inference
summary(mod3_explained_fixed)
