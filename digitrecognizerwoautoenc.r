# Prepare environment
install.packages("tidyverse")
install.packages("DataExplorer")
install.packages("keras")
install.packages("h2o")

# Load packages
library(tidyverse)
library(DataExplorer)
library(keras)
library(h2o)

# List files (extract from samples.zip)
list.files(path = "../input")

# Initiate H2O
h2o.init()

# Importing files
train <- h2o.importFile("..//input/train.csv", destination_frame = "train.hex")

h2o.getId(train)

# May be described in two ways as follows
### h2o.describe(h2o.getFrame("train.hex"))
h2o.describe(train)

# Do the same for test.csv
test <- h2o.importFile("..//input/test.csv", destination_frame = "test.hex")
h2o.describe(test)

# Treating the Training Base, remove label
names(train)
setdiff(names(train), "label")
y <- "label"
x <- setdiff(names(train), y)
train[,y] <- h2o.asfactor(train[,y])
h2o.describe(train)

#Try without autoencode (optimize hyperparameters (Tuning))
splits <- h2o.splitFrame(train, ratios = 0.8, seed = 1)
print(paste("Size of split one: ", (dim(splits[[1]]))[1]))
print(paste("Size of split two: ", (dim(splits[[2]]))[1]))
print(paste("Proportion between splits: ", (dim(splits[[1]])/(dim(splits[[1]])+dim(splits[[2]])))[1]))
h2o.runif(train, 1)
sum((h2o.runif(train, 1) >= 0) & (h2o.runif(train, 1) <= 0.8))
sum((h2o.runif(train, 1) > 0.8) & (h2o.runif(train, 1) <= 1))

# Cartesian strategy 
activation_opt <- c("Rectifier", "Maxout", "Tanh")
l1_opt <- c(0, 0.00001, 0.0001, 0.001, 0.01)
l2_opt <- c(0, 0.00001, 0.0001, 0.001, 0.01)
hyper_params <- list(activation = activation_opt, l1 = l1_opt, l2 = l2_opt)
search_criteria <- list(strategy = "Cartesian")

# Test models
dl_grid <- h2o.grid("deeplearning", x = x, y = y,
                    grid_id = "dl_grid",
                    training_frame = splits[[1]],
                    validation_frame = splits[[2]],
                    distribution = "multinomial",
                    hidden = c(20,20),
                    hyper_params = hyper_params,
                    search_criteria = search_criteria)
print(dl_grid)
dl_gridperf <- h2o.getGrid(grid_id = "dl_grid", decreasing = TRUE)
print(dl_gridperf)
best_dl_model_id <- dl_gridperf@model_ids[[1]]
best_dl <- h2o.getModel(best_dl_model_id)
best_dl
pred_h2o <- h2o.predict(object = best_dl, newdata = test)
pred_h2o
pred <- as.data.frame(pred_h2o$predict)
pred
sample <- read.csv("..//input/sample_submission.csv")
sample
df <- data.frame(sample)
df[,2] <- pred
df
write.csv(df,"..//working/My_Kaggle_h2o.csv", row.names=FALSE)
read.csv("..//working/My_Kaggle_h2o.csv")
