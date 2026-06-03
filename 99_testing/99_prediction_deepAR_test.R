library(tidymodels)
library(modeltime)
library(modeltime.gluonts)
library(modeltime.ensemble)

# Parallel Processing
library(doFuture)

# Core 
library(tidyverse)
library(timetk)
library(skimr)
library(fs)

reticulate::py_discover_config()

# read data ----
full_data <- readRDS("C:/Users/sdben/OneDrive/Professional/Training/R/XX - Practice/04-Financial/quant_trading_R/01_save_data/2025-08-11_full_data.rds")

60 <- 21

# * Data Prepared ----

data_prepared <- full_data %>% 
    filter(!is.na(Return_fwd_21)) %>% 
    drop_na()

data_prepared %>% skim()


# * Future Data ----

future <- full_data %>% 
    filter(is.na(Return_fwd_21))

# future %>% skim()

# future %>% filter(is.na(.view))

# 2.0 TIME SPLIT ----

split <- data_prepared %>% 
    time_series_split(
        date_var = date,
        #assess = 60,
        initial = "3 years",
        assess  = "60 days",
        cumulative = F
    )

split %>% 
    tk_time_series_cv_plan() %>% 
    plot_time_series_cv_plan(date, Return_fwd_21)

training(split) %>% 
    group_by(symbol) %>% 
    plot_time_series(
        date, Return_fwd_21, .facet_ncol = 4
    )

# 3.0 GLUONTS MODELS ----

# * GLUON Recipe Specification ----

recipe <- recipe(Return_fwd_21 ~ symbol + date + rowid, data = training(split)) %>% 
    update_role(rowid, new_role = 'indicator')


# * DeepAR Estimator ----

# Model 1: Default GluonTS
# 3 epochs, 50/epoch, scale = T seems to be best configuration
# prediction length should equal prediction length from training(splits)

model_deepar_1 <- deep_ar(
    id                    = "symbol",
    freq                  = 'B',
    prediction_length     = 60,
    epochs                = 3,
    num_batches_per_epoch = 50,
    scale                 = T
) %>% set_engine('gluonts_deepar')

set.seed(123)
fit_deepar_1 <- workflow() %>% 
    add_model(model_deepar_1) %>% 
    add_recipe(recipe) %>% 
    fit(training(split))

# Model 2: 

model_deepar_2 <- deep_ar(
    id                    = "symbol",
    freq                  = 'B',
    prediction_length     = 60,
    epochs                = 2,
    num_batches_per_epoch = 50
) %>% set_engine('gluonts_deepar')

set.seed(123)
fit_deepar_2 <- workflow() %>% 
    add_model(model_deepar_2) %>% 
    add_recipe(recipe) %>% 
    fit(training(split))


# Model 3: Increase Epochs, Adjust Num Batches Per Epoch, & Add Scaling 
model_deepar_3 <- deep_ar(
    id                    = "symbol",
    freq                  = 'B',
    prediction_length     = 60,
    epochs                = 4,
    num_batches_per_epoch = 50,
    scale                 = T
) %>% set_engine('gluonts_deepar')

set.seed(123)
fit_deepar_3 <- workflow() %>% 
    add_model(model_deepar_3) %>% 
    add_recipe(recipe) %>% 
    fit(training(split))

# * N-BEATS Estimator ----


# Model 4: N-BEATS Default

model_spec_nbeats_4 <- nbeats(
    id                = "symbol",
    freq              = "B",
    prediction_length = 60,
    
    lookback_length   = 2 * 60
) %>%
    set_engine("gluonts_nbeats")

fit_nbeats_4 <- workflow() %>%
    add_model(model_spec_nbeats_4) %>%
    add_recipe(recipe) %>%
    fit(training(split))

# Model 5: Loss Function MASE, Reduce Epochs 3

model_spec_nbeats_5 <- nbeats(
    id                = "symbol",
    freq              = "B",
    prediction_length = 60,
    lookback_length   = 2 * 60,
    epochs            = 4,
    num_batches_per_epoch = 30,
    loss_function     = "MASE"
) %>%
    set_engine("gluonts_nbeats")

fit_nbeats_5 <- workflow() %>%
    add_model(model_spec_nbeats_5) %>%
    add_recipe(recipe) %>%
    fit(training(split))

# Model 6: Model 5 Start, Ensemble

model_spec_nbeats_6 <- nbeats(
    id                    = "symbol",
    freq                  = "B",
    prediction_length     = 60,
    lookback_length       = c(60, 2 * 60),
    epochs                = 4,
    num_batches_per_epoch = 30,
    loss_function         = "MASE",
    bagging_size          = 1
) %>%
    set_engine("gluonts_nbeats_ensemble")

fit_nbeats_6 <- workflow() %>%
    add_model(model_spec_nbeats_6) %>%
    add_recipe(recipe) %>%
    fit(training(split))


# Additional gluonts algorithms ----

model_spec_deepstate_7 <- deep_state(
    id                    = "symbol",
    freq                  = "B",
    prediction_length     = 60,
    epochs                = 2,
    num_batches_per_epoch = 50
    ) %>%
    set_engine("gluonts_deepstate")

set.seed(123)
fit_deepstate_7 <- workflow() %>%
    add_model(model_spec_deepstate_7) %>%
    add_recipe(recipe) %>%
    fit(training(split))

set.seed(123)
model_spec_gpforecast_8 <- gp_forecaster(
    id                    = "symbol",
    freq                  = "B",
    prediction_length     = 60
) %>%
    set_engine("gluonts_gp_forecaster")

fit_gpforecast_8 <- workflow() %>%
    add_model(model_spec_gpforecast_8) %>%
    add_recipe(recipe) %>%
    fit(training(split))

# calibrate and test ----
models_tbl <- modeltime_table(
    fit_deepar_1
    ,fit_deepar_2
    ,fit_deepar_3
    # ,fit_nbeats_4
     ,fit_nbeats_5
    # ,fit_nbeats_6
    ,fit_deepstate_7
    # ,fit_gpforecast_8
)


# Forecast Accuracy
models_tbl %>% 
    modeltime_accuracy(testing(split))


# Forecast Visualization
test_symbols <- data_prepared %>% distinct(symbol) %>% slice_sample(n = 25) %>% pull(symbol)

forecast_test <- models_tbl %>% 
    modeltime_forecast(new_data = testing(split) %>% filter(symbol %in% test_symbols),
                       actual_data = data_prepared %>% filter(symbol %in% test_symbols),
                       keep_data = T) 

#forecast_test %>% skim()

forecast_test %>% filter(is.na(.value)) %>% count(date,.model_id)
forecast_test %>% filter(date == '2025-07-10' & is.na(.value))
testing(split) %>% filter(date == '2025-06-23')

forecast_test %>% 
    #filter(symbol %in% test_symbols) %>% #count(symbol)
    group_by(symbol) %>% 
    plot_modeltime_forecast(.conf_interval_show = F,
                            #.facet_ncol = 2,
                            .trelliscope = T)
