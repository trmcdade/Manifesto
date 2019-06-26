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

mp_setapikey("C:/Users/jacobsm17/R Files/CSV Data/manifesto_apikey.txt") 

full_data <- mp_maindataset(version = "current")

my_corpus <- mp_corpus(countryname == "Ireland" )

tidied_corpus <- my_corpus %>% tidy() 
tidied_corpus

tidy_df <- tidied_corpus %>% 
  unnest_tokens(word,text) 

tidy_df %>% select(manifesto_id, word)

tidy_without_stopwords <- tidy_df %>% 
  anti_join(get_stopwords()) %>% 
  filter(is.na(as.numeric(word))) 

tidy_news_sparse <- tidy_without_stopwords %>%
count(manifesto_id, word) %>%
  cast_sparse(manifesto_id, word, n)


many_models <- data_frame(K = c(20, 30, 40, 50)) %>%
  mutate(topic_model = future_map(K, ~stm(tidy_news_sparse, K = .,
                                          verbose = FALSE)))

heldout <- make.heldout(tidy_news_sparse)

k_result <- many_models %>%
  mutate(exclusivity = map(topic_model, exclusivity),
         semantic_coherence = map(topic_model, semanticCoherence, tidy_news_sparse),
         eval_heldout = map(topic_model, eval.heldout, heldout$missing),
         residual = map(topic_model, checkResiduals, tidy_news_sparse),
         bound =  map_dbl(topic_model, function(x) max(x$convergence$bound)),
         lfact = map_dbl(topic_model, function(x) lfactorial(x$settings$dim$K)),
         lbound = bound + lfact,
         iterations = map_dbl(topic_model, function(x) length(x$convergence$bound)))

k_result


k_result %>%
  transmute(K,
            `Lower bound` = lbound,
            Residuals = map_dbl(residual, "dispersion"),
            `Semantic coherence` = map_dbl(semantic_coherence, mean),
            `Held-out likelihood` = map_dbl(eval_heldout, "expected.heldout")) %>%
  gather(Metric, Value, -K) %>%
  ggplot(aes(K, Value, color = Metric)) +
  geom_line(size = 1.5, alpha = 0.7, show.legend = FALSE) +
  facet_wrap(~Metric, scales = "free_y") +
  labs(x = "K (number of topics)",
       y = NULL,
       title = "Model diagnostics by number of topics",
       subtitle = "These diagnostics indicate that a good number of topics would be around 40")

k_result %>%
  select(K, exclusivity, semantic_coherence) %>%
  filter(K %in% c(20, 40, 60)) %>%
  unnest() %>%
  mutate(K = as.factor(K)) %>%
  ggplot(aes(semantic_coherence, exclusivity, color = K)) +
  geom_point(size = 2, alpha = 0.7) +
  labs(x = "Semantic coherence",
       y = "Exclusivity",
       title = "Comparing exclusivity and semantic coherence",
       subtitle = "Models with fewer topics have higher semantic coherence for more topics, but lower exclusivity")

topic_model <- k_result %>% 
  filter(K == 40) %>% 
  pull(topic_model) %>% 
  .[[1]]

topic_model

td_beta <- tidy(topic_model)

td_beta

td_gamma <- tidy(topic_model, matrix = "gamma",
                 document_names = rownames(tidy_news_sparse))

td_gamma

top_terms <- td_beta %>%  
  arrange(beta) %>% 
  group_by(topic) %>% 
  top_n(7, beta) %>% 
  arrange(-beta) %>% 
  select(topic, term) %>% 
  summarise(terms = list(term))  %>% 
  mutate(terms = map(terms, paste, collapse = ", ")) %>% 
  unnest()

gamma_terms <- td_gamma %>%
  group_by(topic) %>%
  summarise(gamma = mean(gamma)) %>%
  arrange(desc(gamma)) %>%
  left_join(top_terms, by = "topic") %>%
  mutate(topic = paste0("Topic ", topic),
         topic = reorder(topic, gamma))

gamma_terms %>%
  top_n(20, gamma) %>%
  ggplot(aes(topic, gamma, label = terms, fill = topic)) +
  geom_col(show.legend = FALSE) +
  geom_text(hjust = 0, nudge_y = 0.0005, size = 3,
            family = "IBMPlexSans") +
  coord_flip() +
  scale_y_continuous(expand = c(0,0),
                     limits = c(0, 0.09),
                     labels = percent_format()) +
  theme_tufte(base_family = "IBMPlexSans", ticks = FALSE) +
  theme(plot.title = element_text(size = 16,
                                  family="IBMPlexSans-Bold"),
        plot.subtitle = element_text(size = 13)) +
  labs(x = NULL, y = expression(gamma),
       title = "Top 20 topics by prevalence in the Irish Manifesto Coded Data",
       subtitle = "With the top words that contribute to each topic")

gamma_terms %>%
  select(topic, gamma, terms) %>% arrange(desc(gamma)) %>% 
  kable(digits = 3, 
        col.names = c("Topic", "Expected topic proportion", "Top 7 terms")) %>%
  kable_styling(bootstrap_options = "striped", full_width = F)

