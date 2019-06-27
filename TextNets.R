library(manifestoR)
library(tm)
library(tidytext)
library(qdap)
library(topicmodels)
library(tidytext)
library(dplyr)
library(ggplot2)
library(stm)
library(furrr)
library(tidyr)
library(ggthemes)
library(tidyverse)
library(scales)
library(kableExtra)
library(gridExtra)
library(knitr)
library(devtools)
#install.packages("remotes")
#remotes::install_github("cbail/textnets")
library(textnets)
library(dplyr)
library(udpipe)
library(udpipe)
dl <- udpipe_download_model(language = "english")


mp_setapikey("C:/Users/jacobsm17/R Files/CSV Data/manifesto_apikey.txt") 

full_data <- mp_maindataset(version = "current")

countries <- c("Ireland" , "UK" , "Australia", "South Africa", "New Zealand")

my_corpus <- mp_corpus(countryname == countries)

tidy_corpus <- my_corpus %>% tidy() 

tidy_df <- tidy_corpus %>% 
  unnest_tokens(word,text) %>% 
  select(id,party,word)

tidy_def <- tidy_df %>% group_by(id)

prepped_ireland <- PrepText(tidy_def, groupvar = "party", textvar = "word", node_type = "groups", tokenizer = "words", pos = "noun", remove_stop_words = TRUE, strip_numeric = TRUE)

text_network <- CreateTextnet(prepped_ireland)

VisTextNet(text_network, label_degree_cut = 0)