# test catboost directly without tidymodels
library(tidyverse)
library(catboost)
library(data.table)
library(tidymodels)

# data_prepared_dt %>% slice_head(prop = 0.8)
# 
# features   <- training(splits) %>% select(-date,-rowid, -Return_fwd_21, -Return_fwd_10, -Return_fwd_5) %>% mutate(symbol = as.factor(symbol))
# labels     <- training(splits) %>% select(Return_fwd_21)
# train_pool <- catboost.load_pool(data = features, label = labels)
# 
# features_v   <- testing(splits) %>% select(-date,-rowid, -Return_fwd_21, -Return_fwd_10, -Return_fwd_5) %>% mutate(symbol = as.factor(symbol))
# labels_v     <- testing(splits) %>% select(Return_fwd_21)
# valid_pool   <- catboost.load_pool(data = features_v, label = labels_v)

options(scipen = 9999)

# add fourier lags - skip these since may not be helpful
data_prepared_dt <- prices_features_dt[!is.na(Close_momentum_21_252_126) &
                                           !is.na(Vol_WAP_norm),
                                       select(.SD,
                                              -(open:adjusted),
                                              -Return_fwd_5, -Return_fwd_10,
                                              -contains("_lag_"),
                                              -contains("_lead_"))] %>% 
    group_by(symbol) %>%
    tk_augment_fourier(date, .periods = c(252), .K = 1) %>%
    ungroup() %>% 
    setDT()

data <- copy(data_prepared_dt_filter)

setorderv(data, c("date","symbol"))

# train <- data %>% mutate(symbol = as.factor(symbol)) %>% select(-date,-rowid) %>% slice_head(prop = 0.8)
# valid <- data %>% mutate(symbol = as.factor(symbol)) %>% select(-date,-rowid) %>% slice_tail(prop = 0.2)

test_date <- today() - years(2)#-months(6)

set.seed(101)
# split <- initial_split(data[,!c("rowid")], prop = 0.8)
# split <- initial_split(data[date >= test_date,!c("rowid")], prop = 0.8)
split <- initial_split(data[date >= test_date,], prop = 0.8, strata = symbol)
#train <- data %>% slice_sample(prop = 0.8) %>% select(-rowid)
#valid <- data %>% slice_sample(prop = 0.2) %>% select(-rowid)
train <- training(split)
valid <- testing(split)

mean(train$Return_fwd_21)
mean(valid$Return_fwd_21)

summary(train$date)
summary(valid$date)

train %>% count(symbol)
valid %>% count(symbol)

train_pool <- catboost.load_pool(data = train %>% select(-Return_fwd_21,-date, -symbol, -rowid), label = train$Return_fwd_21)
#train_s_pool <- catboost.load_pool(data = train_short %>% select(-Return_fwd_21,-date, -symbol, -rowid), label = train_short$Return_fwd_21)
valid_pool <- catboost.load_pool(data = valid %>% select(-Return_fwd_21,-date, -symbol, -rowid), label = valid$Return_fwd_21)

# modeling
set.seed(121)
start <- Sys.time()
model  <- catboost.train(train_pool,  valid_pool,
                        params = list(loss_function = 'RMSE',
                                      iterations    = 4500
                                      , early_stopping_rounds = 20
                                      #, od_type       = 'Iter'   # Early stopping
                                      #, od_wait       = 20
                                      ,verbose = 50
                                      ))
end <- Sys.time()
end-start

# visualize feature importance
feat_imp <- catboost.get_feature_importance(model) %>% 
    as_tibble(rownames = "Feature") %>% 
    rename(Importance = V1) %>% 
    mutate(Feature = fct_reorder(Feature, Importance,.desc = F))

feat_imp %>% 
    arrange(desc(Importance))

ggplot(feat_imp, aes(y = Feature, x = Importance)) +
    geom_bar(stat='identity')
    #theme(axis.text.x= element_text(angle = 45, hjust = 1)) 

preds     <- predict(model, valid_pool)
predicted <- bind_cols(valid, pred = preds)

rmse(predicted, truth = Return_fwd_21, estimate = pred)
rsq(predicted, truth = Return_fwd_21, estimate = pred)

acc_by_symbol <- predicted[,.(symbol,error = pred - Return_fwd_21)][,.(rmse = sqrt(mean(error^2))), keyby = symbol]

predicted %>% 
    filter(symbol == "MMM") %>% 
    select(date, Return_fwd_21, pred, SAR) %>% 
    pivot_longer(-date) %>% 
    plot_ly(x = ~date, y = ~value, color = ~name, type = "scatter", mode = "lines", colors = c("darkgreen","red4","deepskyblue"))

# testing forecast
forecast_pool <- catboost.load_pool(data = forecast_dt[,!c("date","rowid", "symbol","Return_fwd_21")], label = forecast_dt$Return_fwd_21)

fcst      <- predict(model, forecast_pool)
forecast  <- bind_cols(forecast_dt, predicted = fcst)

forecast[date == max(date),.(symbol, date, predicted)]  %>% 
    left_join(acc_by_symbol) %>% 
    # filter(rmse < 0.05)
    slice_min(rmse, n = 100) %>% 
    arrange(desc(predicted))

cb_fcst_get <- forecast[date == max(date),.(symbol, date, predicted)]  %>% 
    left_join(acc_by_symbol) %>% 
    # filter(rmse < 0.05)
    slice_min(rmse, n = 100) %>% 
    slice_max(predicted, n = 10) %>% 
    pull(symbol)

# tidymodels with bonsai
library(bonsai)

set.seed(121)
start <- Sys.time()
bonsai_fit_catboost <- boost_tree("regression",
                                  trees  = 3500
                                  #,tree_depth = 10
                                  #,learn_rate = 0.01
                                  #,mtry = 0.5
                                  #,stop_iter = 20
                                  ) %>% 
    set_engine('catboost'  
               , eval_metric           = "RMSE"
               , thread_count          = parallelly::availableCores(omit = 1)
               , early_stopping_rounds = 20
               , verbose               = 50
               # , od_type       = 'Iter'   # Early stopping
               # , od_wait       = 20
               # boosting_type = "Plain"
               #,counts = FALSE
    ) %>% 
    fit(Return_fwd_21 ~ ., train %>% select(-symbol, -date))
end <- Sys.time()
end-start

# test with recipe
model_spec_catboost <- boost_tree("regression",
                                  trees  = 4000
                                  #,tree_depth = 5
                                  #,min_n = 20
                                  #,mtry = 5
) %>% 
    set_engine('catboost'
               , early_stopping_rounds = 20
               , thread_count = 6) 

wflw_spec_catboost <- workflow() %>% 
    add_model(model_spec_catboost) %>% 
    add_recipe(recipe_spec %>% step_rm(date))

set.seed(69)
start <- Sys.time()
bonsai_fit_catboost <- wflw_spec_catboost  %>% 
    fit(training(splits))
#fit_resamples(resamples_kfold)
end <- Sys.time()
end-start

# analysis of predictions 
feat_imp <- catboost.get_feature_importance(extract_fit_engine(bonsai_fit_catboost)) %>% 
    as_tibble(rownames = "Feature") %>% 
    rename(Importance = V1) %>% 
    mutate(Feature = fct_reorder(Feature, Importance,.desc = F))

feat_imp %>% plot_ly(x = ~Importance, y = ~Feature, type = "bar", alpha = 0.8)

augment(bonsai_fit_catboost, testing(splits)) %>% rmse(.pred, Return_fwd_21)
augment(bonsai_fit_catboost, valid) %>% rsq(.pred, Return_fwd_21)

bonsai_fcst <- augment(bonsai_fit_catboost, forecast_dt)

bonsai_fcst %>% filter(date == max(date)) %>% select(symbol, .pred) %>% arrange(desc(.pred))

catboost_test_fit <- augment(bonsai_fit_catboost, valid)

catboost_test_fit %>% 
    filter(symbol == "AAPL") %>% 
    select(date,.pred,Return_fwd_21) %>% pivot_longer(-date) %>%  ggplot(aes(x = date,y = value, color = name))+geom_line()

