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

# Load database and train
mnist <- dataset_mnist()

# mnist is a 28x28 image base, here we transform each image into a vector sized 784, each element a pixel
x_train <- mnist$train$x %>% apply(1, as.numeric) %>% t()
x_test <- mnist$test$x %>% apply(1, as.numeric) %>% t()

# Transform images for 0-1 interval so the lose function works properly
x_train <- x_train/255
x_test <- x_test/255

# Try with autoencode
encoding_dim <- 32
input <- layer_input(shape = 784)
encoded <- layer_dense(input, encoding_dim, activation = "relu", 
                       activity_regularizer = regularizer_l1(l = 10e-5))
decoded <- layer_dense(encoded, 784, activation = "sigmoid")
input <- layer_input(shape = 784)
encoded <- layer_dense(input, 128, activation = "relu") %>%
  layer_dense(64, activation = "relu") %>%
  layer_dense(32, activation = "relu")
decoded <- layer_dense(encoded, 64, activation = "relu") %>%
  layer_dense(128, activation = "relu") %>%
  layer_dense(784, activation = "sigmoid")
autoencoder <- keras_model(input, decoded)
autoencoder %>% compile(optimizer='adadelta', loss='binary_crossentropy')

# Estimate parameters in keras
autoencoder %>% fit(
  x_train, x_train,
  epochs = 50,
  batch_size = 256,
  shuffle = TRUE,
  validation_data = list(x_test, x_test)
)

# Autoencoders in H2O
splits_ae <- h2o.splitFrame(train, 0.5, seed = 1)

# first part of the data, without labels for unsupervised learning
train_unsupervised <- splits_ae[[1]]
dim(train_unsupervised)

# second part of the data, with labels for supervised learning
train_supervised <- splits_ae[[2]]
dim(train_supervised)

hidden <- c(256, 128, 64, 128)

# Train the deep learning autoencoder model.
ae_model <- h2o.deeplearning(x = x, 
                             training_frame = train_unsupervised,
                             model_id = "mnist_autoencoder",
                             ignore_const_cols = FALSE,
                             activation = "Tanh",  # Tanh is good for autoencoding
                             hidden = hidden,
                             autoencoder = TRUE)
ae_model

# Next, we can use this pre-trained autoencoder model as a starting point for a supervised deep neural network (DNN).
fit_ae <- h2o.deeplearning(x = x, y = y,
                         training_frame = train_supervised,
                         ignore_const_cols = FALSE,
                         hidden = hidden,
                         pretrained_autoencoder = "mnist_autoencoder")
                         
pred_ae <- h2o.predict(object = fit_ae, newdata = test)
pred_ae   
pred <- as.data.frame(pred_ae$predict)
pred
sample <- read.csv("..//input/sample_submission.csv")
df <- data.frame(sample)
df[,2] <- pred
df
write.csv(df,"..//working/My_autoencoder_h2o.csv", row.names=FALSE)
read.csv("..//working/My_autoencoder_h2o.csv")
fit_ae
