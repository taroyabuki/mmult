library(tidyverse)
read_csv("result.csv") %>%
  ggplot(aes(x=chapter, y=X, color=arch)) +
  geom_point() +
  geom_line() +
  scale_y_log10()
