library(manifestoR)
library(tm)
library(tidytext)
library(dplyr)
library(topicmodels)
library(ggplot2)
library(textnets)

setwd("C:/Users/corea/Documents/SICSS 2019/Group Project")
mp_setapikey("C:/Users/corea/Documents/SICSS 2019/Group Project/manifesto_apikey.txt")

full_data <- mp_maindataset(version = "current")

# The countries
countries <- c("Ireland", "South Africa", "United Kingdom", "Australia","New Zealand")
#countries <- c("South Africa")


#### Country-level manifestos

for (i in 1:length(countries))
{
  print(i)
  print(countries[i])
  
  # Corpus for country i
  country_corpus <- mp_corpus(countryname == countries[i])
  
  tidied_country_corpus <- country_corpus %>% tidy() 
  
  # Get the mapping for party to partyname
  country_partynames <- filter(full_data, countryname== countries[i]) %>%
    select(party, partyname)  %>%
    distinct()
  
  
  tidied_country_corpus_partynames <- left_join(tidied_country_corpus, 
                                                country_partynames, 
                                                by = "party")
  
  
  # Be warned - this takes ~20 min per country
  prepped_country <- PrepText(tidied_country_corpus_partynames, 
                              groupvar = "partyname", 
                              textvar = "text", 
                              node_type = "groups", 
                              tokenizer = "words", 
                              pos = "nouns", 
                              remove_stop_words = TRUE, 
                              compound_nouns = TRUE)
  
  unique(prepped_country$partyname)
  
  country_text_network <- CreateTextnet(prepped_country)
  
  # Loop through a bunch of different alphas to find the lowest one for which all parties display on the graph
  alpha = c(.2, .3, .4, .5, .6, .7, .8, .9, 1) # note that .1 generates an error

  for (j in 1:length(alpha))
  {

    print(j)
    print('graphing now')
    
    file <- paste0(countries[i], "_textnet", alpha[j], ".png")
    title <- paste0("Text Network Results for ", countries[i])
    label <- paste0("alpha=", alpha[j])

    VisTextNet(country_text_network, label_degree_cut = 0, alpha=alpha[j])
    
    last_plot() + labs(title = title, caption = label)
  
    ggsave(filename = file)
  
  }
}

#South Africa - alpha 0.4




#### Party Ideology Manifestos

# Socialist (parfam=20) + Nationalist (parfam=70) parties have smallest frequencies

ideology_corpus <- mp_corpus(parfam == 70 & countryname == "United Kingdom" | countryname == "Australia")
countries <- c("United Kingdom", "Australia")

tidied_ideology_corpus <- ideology_corpus %>% tidy() 

partylist = list()

for (i in 1:length(countries))
{
  # Get the mapping for party to partyname
  country_partynames <- filter(full_data, countryname== countries[i]) %>%
    select(party, partyname)  %>%
    distinct()
  partylist[[i]] <- country_partynames
}

all_partynames <- bind_rows(partylist)
#might have dups in party column with different spellings in partyname -> should remove

tidied_ideology_corpus_partynames <- left_join(tidied_ideology_corpus, 
                                               all_partynames, 
                                               by = "party")

prepped_ideology <- PrepText(tidied_ideology_corpus_partynames, 
                             groupvar = "partyname", 
                             textvar = "text", 
                             node_type = "groups", 
                             tokenizer = "words", 
                             pos = "nouns", 
                             remove_stop_words = TRUE, 
                             compound_nouns = TRUE)

ideology_text_network <- CreateTextnet(prepped_ideology)

print('graphing now')

file <- paste0("conservative_textnet.png")
title <- paste0("Text Network Results for Conservative Parties")

VisTextNet(ideology_text_network, label_degree_cut = 0)

last_plot() + labs(title = title)

ggsave(filename = file)