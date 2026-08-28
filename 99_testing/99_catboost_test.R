# test catboost directly without tidymodels
library(tidyverse)
library(catboost)
library(data.table)
library(tidymodels)
library(timetk)
library(plotly)

# data_prepared_dt %>% slice_head(prop = 0.8)
# 
# features   <- training(splits) %>% select(-date,-rowid, -Return_fwd_21, -Return_fwd_10, -Return_fwd_5) %>% mutate(symbol = as.factor(symbol))
# labels     <- training(splits) %>% select(Return_fwd_21)
# train_pool <- catboost.load_pool(data = features, label = labels)
# 
# features_v   <- testing(splits) %>% select(-date,-rowid, -Return_fwd_21, -Return_fwd_10, -Return_fwd_5) %>% mutate(symbol = as.factor(symbol))
# labels_v     <- testing(splits) %>% select(Return_fwd_21)
# valid_pool   <- catboost.load_pool(data = features_v, label = labels_v)

options(scipen = 99)

# add fourier lags - skip these since may not be helpful
data_pre <- prices_features_dt[!is.na(Close_macd_long_signal_trend),
                                       select(.SD,
                                              -(open:adjusted),
                                              -Return_fwd_5, -Return_fwd_10,
                                              -contains("_lag_"),
                                              -contains("_lead_"))] %>% 
    group_by(symbol) %>%
    tk_augment_fourier(date, .periods = c(252), .K = 2) %>%
    ungroup() %>% 
    # tk_augment_timeseries_signature(date) %>% 
    # select(-matches("(.xts$)|(.iso$)|(.lbl$)|(.hour)|(.minute)|(.second)|(.am.pm)")) %>% 
    # select(-index.num, -diff, -year) %>% 
    setDT()

# explicitly set weeks so that the first week of the year is continuous with the last week of the previous year
options(datatable.week = 'legacy')

data_pre[,":=" (mday = mday(date)
                ,qday = qday(date)
                ,yday = yday(date)
                ,week = week(date)
)]

# data_pre[,":=" (mday = NULL
#                 ,qday = NULL
#                 ,yday = NULL
#                 ,week = NULL
# )]

setorderv(data_pre, c("date","symbol"))

# filter the data for the test-validation and forecating splits
data     <- data_pre[!is.na(Return_fwd_21)]
forecast <- data_pre[is.na(Return_fwd_21)]

# train <- data %>% mutate(symbol = as.factor(symbol)) %>% select(-date,-rowid) %>% slice_head(prop = 0.8)
# valid <- data %>% mutate(symbol = as.factor(symbol)) %>% select(-date,-rowid) %>% slice_tail(prop = 0.2)

test_date <- max(data$date) - months(3)

set.seed(101)
# split <- initial_split(data[,!c("rowid")], prop = 0.8)
# split <- initial_split(data[date >= test_date,!c("rowid")], prop = 0.8)
split <- initial_split(data[date < test_date,], prop = 0.8, strata = symbol)
#train <- data %>% slice_sample(prop = 0.8) %>% select(-rowid)
#valid <- data %>% slice_sample(prop = 0.2) %>% select(-rowid)
train <- training(split)
test <- testing(split)

valid <- data[date >= test_date,]

mean(train$Return_fwd_21)
mean(test$Return_fwd_21)
mean(valid$Return_fwd_21)

summary(train$date)
summary(valid$date)

train %>% count(symbol)
test %>% count(symbol)
valid %>% count(symbol)

train_pool <- catboost.load_pool(data = train %>% select(-Return_fwd_21,-date, -symbol), label = train$Return_fwd_21)
#train_s_pool <- catboost.load_pool(data = train_short %>% select(-Return_fwd_21,-date, -symbol, -rowid), label = train_short$Return_fwd_21)
test_pool <- catboost.load_pool(data = test %>% select(-Return_fwd_21,-date, -symbol), label = test$Return_fwd_21)
valid_pool <- catboost.load_pool(data = valid %>% select(-Return_fwd_21,-date, -symbol), label = valid$Return_fwd_21)

# modeling
set.seed(121)
start <- Sys.time()
model  <- catboost.train(train_pool,  test_pool,
                        params = list(loss_function = 'RMSE',
                                      iterations    = 4000
                                      , early_stopping_rounds = 20
                                      #, od_type       = 'Iter'   # Early stopping
                                      #, od_wait       = 20
                                      ,verbose = 500
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

# shap values
shap <- catboost.get_feature_importance(model, pool = valid_pool,
                                type = "ShapValues")

colnames(shap) <- c(names(valid)[-(1:2)][-49],"expected_value")

nrow(shap)

as_tibble(shap) |>
    pivot_longer(-expected_value) |> 
    summarise(mean_shap = mean(abs(value)),
              .by = name) |> 
    mutate(name = fct_reorder(name,mean_shap)) |> 
    ggplot(aes(x = mean_shap, y = name))+
    geom_col()

as_tibble(shap) |>
    slice_sample(prop = 0.4) |> 
    pivot_longer(-expected_value) |> 
    ggplot(aes(x = value, y = expected_value))+
    geom_point()+
    facet_wrap(~name, scales = "free_y")

# test set accuracy
preds     <- predict(model, test_pool)
predicted <- bind_cols(test, pred = preds)
metrics(predicted, truth = Return_fwd_21, estimate = pred)

# validate out of sample prediction accuracy
preds     <- predict(model, valid_pool)
predicted <- bind_cols(valid, pred = preds)
metrics(predicted, truth = Return_fwd_21, estimate = pred)

# accuracy by symbol
acc_by_symbol <- predicted[,.(symbol,error = pred - Return_fwd_21)][,.(rmse = sqrt(mean(error^2))), keyby = symbol]

predicted %>% 
    filter(symbol == "MMM") %>% 
    select(date, Return_fwd_21, pred) %>% 
    pivot_longer(-date) %>% 
    plot_ly(x = ~date, y = ~value, color = ~name, type = "scatter", mode = "lines", colors = c("darkgreen","red4","deepskyblue"))

# testing forecast
forecast_pool <- catboost.load_pool(data = forecast[,!c("date","symbol","Return_fwd_21")], label = forecast$Return_fwd_21)

fcst      <- predict(model, forecast_pool)
forecast  <- bind_cols(forecast, predicted = fcst)

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


# test model interpretation with IML
library(iml)

predict_function <- function(model, newdata){
        preds <- predict(model, catboost.load_pool(newdata))
        as.data.frame(preds)
}

x <- test[,select(.SD,-Return_fwd_21,-date, -symbol)]
predictor <- Predictor$new(model, data = x, y = test$Return_fwd_21,
                           predict.function = predict_function)

imp <- FeatureImp$new(predictor, loss = "rmse")
plot(imp)

ale <- FeatureEffect$new(predictor, feature = "Close_252_min_diff", grid.size = 10)
ale$plot()

pdp <- FeatureEffect$new(predictor, feature = "Close_252_min_diff", method = "pdp")
plot(pdp)
