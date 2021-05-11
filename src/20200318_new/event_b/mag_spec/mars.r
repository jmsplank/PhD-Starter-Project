#! /usr/local/bin/Rscript

wd <- getwd()
data <- read.csv(file = '/Users/jamesplank/Documents/PHD/PhD-Starter-Project/src/20200318_new/event_b/mag_spec/raw_r.csv')
library(earth)
mars <- earth(x = data['x'], y = data['y'], keepxy = TRUE)
prediction <- predict(mars)
write.csv(prediction, paste(wd, 'src/20200318_new/event_b/mag_spec', 'mars.csv', sep='/'), row.names = FALSE)
