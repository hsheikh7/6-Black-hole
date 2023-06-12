

# LIME FEATURE EXPLANATION ----

# Setup 
# Load Libraries 
library(h2o)
library(recipes)
library(readxl)
library(tidyverse)
library(tidyquant)
library(caret)
library(lime)

# Load Data
dataset <- read.csv("data/product_backorders.csv") 
glimpse(dataset)

# Step 1: Split the Dataset into train and test sets
# Set the seed for reproducibility
set.seed(123)

# Specify the response variable
response <- "went_on_backorder"

train_indices <- createDataPartition(dataset[, response], p = 0.7, list = FALSE)
train_data <- dataset[train_indices, ]
test_data <- dataset[-train_indices, ]

# dimensions of the train and test sets
dim(train_data)
dim(test_data)

# Specify the response and predictor variables
response_var <- "went_on_backorder"
predictor_vars <- setdiff(colnames(train_data), response_var)

# Step 2: ML Preprocessing Recipe 
recipe_obj <- recipe( went_on_backorder ~ ., data = train_data) %>%
  step_zv(all_predictors()) %>%
  #step_mutate_at(c("JobLevel", "StockOptionLevel"), fn = as.factor) %>% 
  prep()

recipe_obj

train_tbl <- bake(recipe_obj, new_data = train_data)
test_tbl  <- bake(recipe_obj, new_data = test_data)

# Step 3. Models 
h2o.no_progress()
h2o.init()

automl_leader <- h2o.loadModel("model/StackedEnsemble_AllModels_1_AutoML_17_20230610_222745")
automl_leader


# Step 4. LIME Starter 
# Making Predictions 

predictions_tbl <- automl_leader %>% 
  h2o.predict(newdata = as.h2o(test_data)) %>%
  as.tibble() %>%
  bind_cols(
    test_tbl %>%
      select( went_on_backorder, sku)
  )

predictions_tbl

test_tbl %>%
  slice(1) %>%
  glimpse()


# Step 5: Single Explanation ----

explainer <- train_data %>%
  select(-went_on_backorder) %>%
  lime(
    model           = automl_leader,
    bin_continuous  = TRUE,
    n_bins          = 4,
    quantile_bins   = TRUE
  )

explainer

?lime::explain

explanation <- test_data %>%
  slice(1) %>%
  select(-went_on_backorder) %>%
  lime::explain(
    
    # Pass our explainer object
    explainer = explainer,
    # Because it is a binary classification model: 1
    n_labels   = 1,
    # number of features to be returned
    n_features = 8,
    # number of localized linear models
    n_permutations = 5000,
    # Let's start with 1
    kernel_width   = 1
  )

explanation

explanation %>%
  as.tibble() %>%
  select(feature:prediction) 

g <- plot_features(explanation = explanation, ncol = 1)



# Step 6: Multiple Explanations 

explanation <- test_data %>%
  slice(1:20) %>%
  select(-went_on_backorder) %>%
  lime::explain(
    explainer = explainer,
    n_labels   = 1,
    n_features = 8,
    n_permutations = 5000,
    kernel_width   = 0.5
  )

explanation %>%
  as.tibble()

plot_features(explanation, ncol = 4)

plot_explanations(explanation)
