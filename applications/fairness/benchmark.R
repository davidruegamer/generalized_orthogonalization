library(parallel)
library(fairml)
library(tidyr)
library(dplyr)
library(tibble)
library(ranger)

# wrapper for orthog
source("../../simulations/common/functions.R")

# wrapper for fairml
frrm_sp_komiyama <- function(...) frrm(..., definition = "sp-komiyama")
fgrrm_sp_komiyama <- function(...) fgrrm(..., definition = "sp-komiyama")
frrm_eo_komiyama <- function(...) frrm(..., definition = "eo-komiyama")
fgrrm_eo_komiyama <- function(...) fgrrm(..., definition = "eo-komiyama")
frrm_if_berk <- function(...) frrm(..., definition = "if-berk")
fgrrm_if_berk <- function(...) fgrrm(..., definition = "if-berk")
zlm_wo_extras <- function(..., lambda) zlrm(...)
zlrm_wo_extras <- function(..., family, lambda) zlrm(...)

extract_info_model <- function(mod) rownames_to_column(as.data.frame(summary(mod)$coefficients[,c(1,4)])) %>% 
  pivot_longer(-rowname)

load_data <- function(name){
  get(name, envir=getNamespace("fairml"))
}

data_info <- data.frame(
  names = c("adult", "bank", "communities.and.crime",
            "compas", "drug.consumption", "health.retirement",
            "law.school.admissions", "national.longitudinal.survey",
            "obesity.levels"),
  fams = c("binomial", "binomial", "gaussian", "binomial",
           "multinomial", "poisson", "gaussian", "gaussian",
           "multinomial"),
  prot = c("c(8,9)", "1", "c(4, 93)", "c(8, 10)", "c(2, 3, 4)",
           "c(2, 24, 25, 26)", "c(1, 8)", "c(1, 3)", c("c(1, 2)")),
  resp = c(14, 19, 102, 9, 1, 23, 6, 5, 17)
)

data_info <- data_info[data_info$fams %in% c("binomial", "poisson"),]

choose_methods <- function(fam)
{
  
  list_of_methods <- switch (
    fam,
    gaussian = c("frrm_sp_komiyama", "frrm_eo_komiyama", "frrm_if_berk", "nclm", "zlm_wo_extras"),
    binomial = c("fgrrm_sp_komiyama", "fgrrm_eo_komiyama", "fgrrm_if_berk", "zlrm_wo_extras"),
    poisson = c("fgrrm_sp_komiyama", "fgrrm_eo_komiyama", "fgrrm_if_berk"),
    multinomial = c("fgrrm_sp_komiyama", "fgrrm_eo_komiyama", "fgrrm_if_berk")
  )
  
  return(list_of_methods)
  
}

benchmark_fun <- function(
    dataset, 
    unfairness = 0.05,
    di = data_info
)
{
  
  cat(dataset,"\n")
  # get data and methods
  data <- load_data(dataset)
  # extra treatment
  if(dataset == "drug.consumption") 
    data <- data[,c("Meth", "Age", "Gender", "Race",  
                    "Education", "Nscore", "Escore", "Oscore", "Ascore",
                    "Cscore", "Impulsive", "SS")]
  if(dataset == "health.retirement")
    data <- data[,-ncol(data)]
  if(dataset == "national.longitudinal.survey")
    data <- data[, setdiff(names(data), c("income96", "income06"))]
  if(dataset == "communities.and.crime")
    data <- na.omit(data[, setdiff(names(data), c("state", "county"))])
  if(dataset == "health.retirement")
    data <- na.omit(data)
  if(dataset == "adult"){ # fairml seem to not be able to handle more data
    set.seed(32)
    data <- data[sample(1:nrow(data), pmin(nrow(data), 5000)),]
    data <- droplevels(data)
  }
  info <- di[di$name == dataset,]
  family <- info$fams
  models <- choose_methods(family)
  
  resp <- data[, info$resp]
  sens_ids <- eval(parse(text=info$prot))
  sens <- data[, sens_ids, drop = FALSE]
  pred_id <- setdiff(1:ncol(data), c(info$resp, sens_ids))
  preds <- data[, pred_id, drop = FALSE]
  
  args <- list(
    response = resp,
    predictors = preds,
    sensitive = sens,
    unfairness = unfairness
  )
  
  if(family != "gaussian")
    args$family <- family
  if(family == "binomial")
    resp <- as.factor(resp)
  # if(dataset == "bank")
  #   args$lambda <- 0.1
  
  results_fair_methods <- lapply(models, function(m) try(do.call(m, args = args)))
  
  stats_fair <- lapply(1:length(models), function(j){ 
    
    m <- results_fair_methods[[j]]
    
    if(inherits(m, "try-error"))
      return(data.frame(method = models[j], dataset = dataset, 
                        rowname = c(NA, NA),
                        name = c("Estimate", "Pr(>|z|)"),
                        value = c(NA, NA)))
    
    yhat <- if(grepl("zlrm", models[j])) predict(m, preds) else
      predict(m, preds, sens)
    
    eval_mod <- glmgen(yhat, sens, fam = family, intercept = T)
  
    return(cbind(method = models[j], dataset = dataset, 
                 rbind(extract_info_model(eval_mod), rf_check(yhat, sens))))
    
  })
  
  result_ortho <- wrapper_orthog(resp, 
                                 cbind(preds, sens), 
                                 sens, 
                                 family = family)
  ortho <- cbind(method = "ortho", dataset = dataset, 
                 rbind(extract_info_model(result_ortho[[1]]),
                       result_ortho[[2]]))
  
  return(do.call("rbind", c(stats_fair, list(ortho))))
  
}

# debug(benchmark_fun)

res <- mclapply(data_info$names, function(name) benchmark_fun(name), mc.cores = 1)

saveRDS(res, file = "fairness_benchmark.RDS")

res <- do.call("rbind", res)

library(ggplot2)
library(dplyr)

my_palette <- c("#E69F00", "#56B4E9", "#009E73", 
                "#0072B2", "#D55E00", "#CC79A7")

res$method <- factor(res$method, levels = unique(res$method),
                     labels = c("Komiyama1", "Komiyama2", "Berk", "Zafar", "Ours"))
res$name <- factor(res$name, levels = unique(res$name), labels = c("effect", "p-value"))

ggplot(res %>% filter(rowname != "(Intercept)") 
         # mutate(
         #   method = factor(method, levels = unique(method), 
         #                 labels = c("",
         #                            "")),
           # name = factor(name, levels = c("coef", "pval"),
           #               labels = c("effect", "p-value")),
           # type = factor(type, levels = unique(type),
           #               labels = c(expression(C^l), expression(C^h))
           # )
         #) %>% filter(abs(value) <= 10) %>% filter(load == ll),  
       , aes(x = dataset, y=value, colour = method)) +
  geom_boxplot() +
  facet_wrap(~ name, scales="free") + 
  theme_bw() + 
  scale_colour_manual(values = my_palette) + 
  theme(
    legend.title = element_blank(),
    text = element_text(size = 16),
    legend.position="bottom"
  )

ggsave(file = "results_fairness_benchmark.pdf", width = 14, height = 4)
