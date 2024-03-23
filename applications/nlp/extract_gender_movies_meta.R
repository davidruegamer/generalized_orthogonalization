library(tidyverse)
library(stringr)

data <- read.csv("credits.csv")

extract_first_gender <- function(json_str) {
  matches <- gregexpr("'gender': ([0-9]+)", json_str)
  first_match <- regmatches(json_str, matches)[[1]][1]
  first_gender <- gsub("'gender': ", "", first_match)
  return(first_gender)
}

gender_main_actor <- as.numeric(sapply(data$cast, extract_first_gender))
gender_director <- as.numeric(sapply(data$crew, extract_first_gender))

saveRDS(data.frame(id = data$id, gender_main_actor = gender_main_actor,
                   gender_director = gender_director), file="sex.RDS")
