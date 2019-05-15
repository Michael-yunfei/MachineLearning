# miscellaneous R code
# @ Michael

getwd()
setwd('/Users/Michael/Documents/MachineLearning/ELS')

prostate <- read.csv('prostate.csv', sep='')
head(prostate)
summary(prostate)
plot(prostate$age, prostate$lweight)

library(tidyverse)
