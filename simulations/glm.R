source("common/functions.R")

res <- sim_function(
  sample_fun = function(n, p) rbinom(n, 1, p),
  h = plogis,
  g = qlogis,
  fam = binomial(), 
  sd_sample = 0.5,
  nrcores = 1,
  which_corr = "Lagrangian"
)

saveRDS(res, file="results/binom.RDS")

pltres <- plot_function(res)

wd <- 9.5
ht <- 4.5

pltres[[1]]

ggsave("glmbin_load0.pdf", width = wd, height = ht)

pltres[[2]]

ggsave("glmbin_load1.pdf", width = wd, height = ht)

pltres[[3]]

ggsave("glmbin_load2.pdf", width = wd, height = ht)
