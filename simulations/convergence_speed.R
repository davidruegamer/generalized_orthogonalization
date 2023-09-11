source("common/functions.R")

res <- sim_function(
  sample_fun = function(n, p) rbinom(n, 1, p),
  h = plogis,
  g = qlogis,
  fam = binomial(), 
  sd_sample = 0.1,
  q_vals = c(10, 50, 100),
  p_vals = c(2, 5, 10),
  n_vals = c(100, 1000, 10000),
  rep_vals = 1,
  load_fac_X_on_Z = c(0, 1 ,2)
)

saveRDS(res, file="results/conv.RDS")

hists <- do.call("rbind", lapply(res, function(x) 
  cbind(x$setting, value = x$hist, iter = 1:length(x$hist))))
hists <- hists %>% group_by(p,q,n,load) %>% 
  summarize(max_iter = max(iter),
            value = value[which.max(iter)]) 

library(tidyverse)

hist(hists$iter, breaks=500)

g1 <- ggplot(hists, aes(x = cut(max_iter, breaks=c(1,10,50,100,999,1000)), y=value/1e-5)) +
  geom_boxplot(varwidth = T) +
  # facet_grid(p*q ~ n*load) + #, scales="free", labeller = labeller(n = label_both)) + 
  theme_bw() + 
  # scale_colour_manual(values = my_palette) + 
  theme(
    #legend.title = element_blank(),
    text = element_text(size = 14),
    #legend.position="bottom"
  ) + xlab("Stopping iteration") + ylab("Norm of difference (divided by 1e-5)")


g2 <- ggplot(hists, aes(x = max_iter)) + 
  geom_histogram(bins = 100) +  
  theme_bw() + theme(
    #legend.title = element_blank(),
    text = element_text(size = 14),
    #legend.position="bottom"
  ) + xlab("Stopping iteration") + ylab("Count")
  
library(gridExtra)

p1 <- grid.arrange(g1,g2, ncol=2)

ggsave(plot = p1, "conv_speed.pdf", width = 10, height = 5)
