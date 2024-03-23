source("common/functions.R")

res <- sim_function(
  sample_fun = function(n, p) rpois(n, p),
  h = exp,
  g = log,
  fam = poisson(),
  nrcores = 1,
  which_corr = "Lagrangian"
)

saveRDS(res, file="results/poisson.RDS")

pltres <- plot_function(res)

wd <- 9.5
ht <- 4.5

pltres[[1]]

ggsave("glmpois_load0.pdf", width = wd, height = ht)

pltres[[2]]

ggsave("glmpois_load1.pdf", width = wd, height = ht)

pltres[[3]]

ggsave("glmpois_load2.pdf", width = wd, height = ht)
