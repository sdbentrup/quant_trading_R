# forecast stock returns
# notes and commentary ----
# try to predict returns 21-days forward. consider papers:
# https://jfin-swufe.springeropen.com/articles/10.1186/s40854-024-00644-0
# https://www.sciencedirect.com/science/article/pii/S2666827025000143
# https://drive.google.com/file/d/1uvjBJ9D09T0_sp7kQppWpD-xelJ0KQhc/edit

# ideas:
# nested forecasting: https://business-science.github.io/modeltime/articles/nested-forecasting.html
# adam regression: https://business-science.github.io/modeltime/reference/adam_reg.html

# 0.0 Setup ----
# * packages ----
# Time Series ML
library(tidymodels)
library(modeltime)
library(modeltime.ensemble)
library(modeltime.resample)
library(finetune)
library(xgboost)
library(lightgbm)
library(prophet)
library(bonsai)
# library(boostime)
# library(finnts)

# Timing & Parallel Processing
library(tictoc)
library(future)
library(future.apply)
library(doFuture)
# library(future.mirai)

# Core 
library(tidyquant)
library(tidyverse)
library(timetk)
library(data.table)
library(plotly)
library(rstatix)
library(tidytable)

# Modeling and stock analytics
library(baguette)
library(PortfolioAnalytics)
library(shapviz)
library(ggthemes)
library(rvest) # for read_html()
library(TTR)
library(xts)
library(shapviz)

# * dates ----
from <- today() - years(4)
testing_symbol <- 'AAPL'

# not used
# test_start <- today() - years(2)

# * Calibrate function ----
calibrate_and_plot <- function(..., type = 'testing', plot = T){
    
    if (type == 'testing'){
        new_data = testing(splits)
    } else {
        new_data = training(splits)
    }
    
    calibration_tbl <- modeltime_table(...) %>% 
        modeltime_calibrate(new_data)
    
    print(calibration_tbl %>% modeltime_accuracy())
    
    if (plot == T){calibration_tbl %>% 
            modeltime_forecast(
                new_data = new_data,
                actual_data = data_prepared_dt_filter
            ) %>% 
            plot_modeltime_forecast(.conf_interval_show = F)
    } else {
        return()
    }
}

# * plot forecast function ----
plot_fcst <- function(forecast, scales = "fixed", ncol = 3){
    g <- forecast %>% 
        group_by(symbol) %>% 
        ggplot(aes(x = date, y = .value, color = .model_desc))+
        geom_line()+
        geom_hline(yintercept = 0)+
        facet_wrap(~symbol, ncol = ncol, scales = scales)+
        scale_x_date(date_breaks = "6 months", date_labels =  "%m-%Y")+
        theme_bw()+
        theme(legend.position = "bottom",
              strip.background = element_rect("#0076BA"),
              strip.text = element_text(color = "white"),
              axis.text.x=element_text(angle=60, hjust=1))+
        scale_color_tq()+
        ylab("")
    
    ggplotly(g)
}

# 1.0 Data ----
# * symbols ----
# SP500
sp500 <- tq_index("SP500") #tq_index() options include "DOW"       "DOWGLOBAL" "SP400"     "SP500"     "SP600" 
# sp400 <- tq_index("SP400")
# sp600 <- tq_index("SP600")

exclude_symbols <- c("TSLA","PLTR")

sp500_symbols <- sp500 %>% 
    filter(symbol != "-" & !str_detect(company,"CL C")) %>% 
    filter(symbol %notin% exclude_symbols) %>% 
    slice_max(weight, n = 250) %>%
    # slice_sample(prop = 0.1) %>%
    arrange(symbol) %>% 
    pull(symbol) 

# sp400_symbols <- sp400 %>% 
#     filter(symbol != "-" & !str_detect(company,"CL C")) %>% 
#     filter(symbol %notin% exclude_symbols) %>% 
#     slice_max(weight, n = 10) %>%
#     # slice_sample(prop = 0.1) %>%
#     arrange(symbol) %>% 
#     pull(symbol) 

# symbols <- union(sp500_symbols, sp400_symbols)
symbols <- sp500_symbols

# * get stock price data ----
prices_base <- tq_get(symbols, from = from) %>% 
    setDT()

# * reduce data with filters ----
# check for any duplicates by date. yahoo can sometimes do this
# create a count of symbols by day, then count each observation within the day
prices_base[,N := .N, keyby = .(symbol,date)][,day_count := 1:.N, keyby = .(symbol,date,N)]

# keep only the unique daily value or the second if there are 2 on a day
prices_base <- prices_base[N == 1 | N == 2 & day_count == 2][,!c("N","day_count")]


# remove cases with less than 4 years of data
symbols_filtered_list <- prices_base[!is.na(close),
                                     .(count = .N,
                                       max = max(date),
                                       min = min(date)),
                                     keyby = symbol][count >= 252*3 & max == max(max),
                                                     symbol]

# remove cases with a most-recent trading price of $10 or less/share
symbols_filtered_list <- symbols_filtered_list[!symbols_filtered_list %in%
                                                   prices_base[date == max(date),.(symbol,close)][close <= 10,symbol]]

# remove cases with too many days of 0 volume
symbols_filtered_list <- symbols_filtered_list[!symbols_filtered_list %in%
                          prices_base[volume == 0,.N, keyby = symbol][N > 1,symbol]]


# * create filtered dataset ----
prices_dt <- prices_base[symbol %in% symbols_filtered_list &
                             !is.na(close),]

# ** view data ----
prices_dt

prices_dt[,.(.N),keyby = symbol] %>% arrange(N)
prices_dt[,(uniqueN(symbol))]

# Engineered indicators ----
# https://blog.elearnmarkets.com/best-25-technical-indicators/

# * function for features ----
add_features <- function(prices_dt, price) {
    price_col <- prices_dt[[price]]
    
    # Calculate Indicators
    prices_dt[, Close_macd_long := MACD(price_col, 50, 200, 30)[, "macd"]]
    prices_dt[, Close_macd_long_signal := MACD(price_col, 50, 200, 30)[, "signal"]]
    prices_dt[, Close_macd_short := MACD(price_col, 18, 26, 18, list(list(EMA, wilder=TRUE),list(EMA, wilder=TRUE),list(EMA, wilder=TRUE)))[, "macd"]]
    prices_dt[, Close_macd_short_signal := MACD(price_col, 18, 26, 18, list(list(EMA, wilder=TRUE),list(EMA, wilder=TRUE),list(EMA, wilder=TRUE)))[, "signal"]]
    prices_dt[, ":=" (
        Close_macd_long_trend = frollmean(Close_macd_long,21),
        Close_macd_long_signal_trend = frollmean(Close_macd_long_signal,21),
        Close_macd_short_trend = frollmean(Close_macd_short,21),
        Close_macd_short_signal_trend = frollmean(Close_macd_short_signal,21),
        Close_macd_long_trading_signal = Close_macd_long-Close_macd_long_signal,
        Close_macd_short_trading_signal = Close_macd_short-Close_macd_short_signal
    )]
    prices_dt[, Close_ema_10_norm  := EMA(price_col, n = 10) / price_col]
    prices_dt[, Close_ema_21_norm  := EMA(price_col, n = 21) / price_col]
    prices_dt[, Close_ppo_line_12_26 := ((EMA(price_col, 12) - EMA(price_col, 26)) / EMA(price_col, 26)) * 100]
    # prices_dt[, Close_oscillator_7_14_28 := ultimateOscillator(prices_dt[, .(high, low, price_col)])]
    prices_dt[, Close_roc_0_1      := ROC(price_col, n = 1)]
    #prices_dt[, Close_roc_0_1_roll := frollmean(Close_roc_0_1, n=21, align = "right", na.rm = T)]
    # prices_dt[, Close_roc_0_5      := ROC(price_col, n = 5)]
    prices_dt[, Close_roc_0_21     := ROC(price_col, n = 21)]
    prices_dt[, Close_roc_0_1_rolling_std_win_63 := frollsd(Close_roc_0_1, 63, fill = NA, align = "right")]
    prices_dt[, Close_natr_63      := ATR(prices_dt[,.(high, low, price_col)], n = 63)[,"atr"]/eval(quote(get(price)))]
    # prices_dt[, Close_rsi_10       := RSI(price_col, n = 10)]
    prices_dt[, Close_rsi_21       := RSI(price_col, n = 21)]
    prices_dt[, Close_cmo_28       := CMO(price_col, n = 28)]
    prices_dt[, Close_cmo_ma       := EMA(Close_cmo_28, n = 21)]
    prices_dt[, Close_cmo_signal   := Close_cmo_28/Close_cmo_ma]
    # prices_dt[, Close_rolling_126_norm := (price_col - frollmean(price_col, 126)) / frollsd(price_col, 126, align = "right")] #used to be standardize vec
    # prices_dt[, Close_rolling_std_126 := frollsd(price_col, 126, align = "right")]
    
    # prices_dt[, Close_SNR_21       := SNR(prices_dt[, .(high, low, price_col)], n = 21)]
    prices_dt[, Close_rel_vol_14   := 100 - 100 / (1 + runSD(price_col, 14))]
    prices_dt[, Close_252_max_diff := price_col/frollmax(price_col, 252, align = 'right')]
    prices_dt[, Close_252_min_diff := price_col/frollmin(price_col, 252, align = 'right')]
    prices_dt[, Close_kst          := KST(price_col)[,"kst"]]
    prices_dt[, Close_kst_signal   := KST(price_col)[,"signal"]]
    prices_dt[, Close_TDI          := TDI(price_col, n = 20, multiple = 2)[,"tdi"]]
    prices_dt[, Close_TDI_di       := TDI(price_col, n = 20, multiple = 2)[,"di"]]
    # prices_dt[, Close_TRIX         := TRIX(price_col, n = 20, nSig = 9, "EMA", percent = TRUE)[,"TRIX"]]
    prices_dt[, Close_TRIX_signal  := TRIX(price_col, n = 20, nSig = 9, "EMA", percent = TRUE)[,"signal"]]
    prices_dt[, SAR                := SAR(prices_dt[, .(high, low)])]
    # prices_dt[, SMI                := SMI(prices_dt[, .(high, low, price_col)])[,"SMI"]]
    prices_dt[, SMI_signal         := SMI(prices_dt[, .(high, low, price_col)])[,"signal"]]
    prices_dt[, CMF                := CMF(prices_dt[, .(high, low, price_col)], volume)]
    # prices_dt[, EMV                := EMV(prices_dt[, .(high, low)], volume)[,"emv"]]
    prices_dt[, maEMV              := EMV(prices_dt[, .(high, low)], volume)[,"maEMV"]]
    
    prices_dt[, Close_ADX_DIp      := ADX(prices_dt[,.(high, low, price_col)])[,"DIp"]]
    prices_dt[, Close_ADX_DIn      := ADX(prices_dt[,.(high, low, price_col)])[,"DIn"]]
    # prices_dt[, Close_ADX_DX       := ADX(prices_dt[,.(high, low, price_col)])[,"DX"]]
    prices_dt[, Close_ADX_ADX      := ADX(prices_dt[,.(high, low, price_col)])[,"ADX"]]
    
    # relative vigor index
    prices_dt[,":=" (numSMA = SMA(price_col - open, n = 10), denSMA = SMA(high - low, n = 10))][,":=" (rvi_signal = SMA(numSMA/denSMA, 4))] # rvi = numSMA/denSMA, 
    prices_dt[,":=" (numSMA = NULL, denSMA = NULL)]
    
    # create indicators for intraday effects and day return
    # remove intraday are low importance intraday = high/low-1,
    prices_dt[, ":=" (HLC_ratio = (high-low)/close,
                      OC_ratio  = (open-close)/close)]
    
    # Volume-based indicators
    # prices_dt[, Vol_ema_21_norm  := EMA(volume, n = 21) / volume]
    prices_dt[, Vol_roc_0_1      := ROC(volume, n = 1)] # leave this one for other calculations
    
    # create indicators based on VWAP
    prices_dt[, Vol_WAP           := VWAP(price_col, volume)] # only used for ratios, then dropped
    prices_dt[, Vol_WAP_log       := log(Vol_WAP)]
    prices_dt[, Vol_WAP_Close     := Vol_WAP/price_col]
    # prices_dt[, Vol_WAP_EMA       := EMA(Vol_WAP, n = 21)] # only used for ratios, then dropped
    # prices_dt[, Vol_WAP_EMA_norm  := standardize_vec(EMA(Vol_WAP, n = 21))]
    # prices_dt[, Vol_WAP_EMA_ratio := Vol_WAP/Vol_WAP_EMA]
    prices_dt[, Vol_WAP_ROC       := ROC(Vol_WAP, n = 21)]
    prices_dt[, Vol_roc_0_1_rolling_std_63 := frollsd(Vol_roc_0_1, 63, align = "right")]
    
    # standardize VWAP by 252 days
    prices_dt[, Vol_WAP_norm      := (Vol_WAP - frollmean(Vol_WAP, n = 252, align = "right"))/frollsd(Vol_WAP, n = 252, align = "right")]
    # prices_dt[, Vol_WAP_roll_ratio := Vol_WAP/Vol_WAP_norm_roll]
    
    prices_dt[, ':=' (Vol_WAP = NULL 
                      # ,Vol_WAP_EMA = NULL
    )]
    
    #prices_dt[, Vol_OBV_standard := standardize_vec(OBV(price_col, volume))]
    prices_dt[, Vol_OBV := OBV(price_col, volume)]
    prices_dt[, Vol_OBV_std := (Vol_OBV - frollmean(Vol_OBV, n = 126, align = 'right'))/frollsd(Vol_OBV, n = 126, align = 'right')]
    prices_dt[, Vol_OBV := NULL]
    
    prices_dt[, Vol_MFI_21 := MFI(prices_dt[,.(high, low, close)], volume, n = 21)]
    
    # Momentum
    prices_dt[, `:=`(
        Close_lag_1   = data.table::shift(price_col, n = 1),
        Close_lag_21  = data.table::shift(price_col, n = 21),
        Close_lag_252 = data.table::shift(price_col, n = 252)
    )]
    
    prices_dt[, Close_momentum_21_252_126 := ((Close_lag_21 / Close_lag_252 - 1) - (Close_lag_1 / Close_lag_21 - 1))/frollsd(Close_roc_0_1, 126, align = "right")]
    
    # Z-score features
    prices_dt[, Close_zscore_126 := (price_col - frollmean(price_col, 126, align = "right")) / frollsd(price_col, 126, align = "right")]
    
    prices_dt[, ":=" (Close_roc_0_1 = NULL, 
                      Vol_roc_0_1   = NULL)]
    
    # Efficiently compute lead values & forward returns without loops
    T <- c(5, 10, 21)
    lead_cols   <- paste0("Close_lead_", T)
    return_cols <- paste0("Return_fwd_", T)
    
    prices_dt[, (lead_cols) := lapply(T, function(t) data.table::shift(price_col, -t))]
    prices_dt[, (return_cols) := lapply(lead_cols, function(col) (get(col) / price_col) - 1)]
    
    return(prices_dt)
}

# * Apply function ----
prices_features_list <- future_lapply(split(prices_dt, by = "symbol"), 
                                      add_features,
                                      price = "close")

# Combine the results back into a `data.table`
prices_features_dt <- rbindlist(prices_features_list, use.names = TRUE, fill = TRUE)

# Save data for future use ----
write_rds(prices_features_dt, str_glue("01_save_data/{today()}_prices_features_dt.rds"))


#rm(list=ls(pattern="^wflw_"))

# 2.0 TIME TEST/TRAIN SPLIT ----
# * clean for splitting ----
# remove NA values, add fourier transform features, set order
setorderv(prices_features_dt, c("symbol","date"))

data_prepared_dt <- setDT(
    prices_features_dt[!is.na(Close_momentum_21_252_126) &
                           !is.na(Vol_WAP_norm),
                       select(.SD,
                              -(open:adjusted),
                              -Return_fwd_5, -Return_fwd_10,
                              -contains("_lag_"),
                              -contains("_lead_"))] %>% 
        tk_augment_fourier(date, .periods = c(63,252), .K = 1)
    )

data_prepared_dt[,rowid := 1:.N]

setcolorder(data_prepared_dt, "rowid", before = "symbol")

# * Split into train and forecast sets ----
forecast_dt <- data_prepared_dt[is.na(Return_fwd_21)]

data_prepared_dt_filter <- data_prepared_dt[!is.na(Return_fwd_21),]

splits <- data_prepared_dt_filter %>% 
    time_series_split(
    date_var   = date,
    initial    = 252 * 2,#"2 years",
    assess     = 21,
    cumulative = F
    )

set.seed(123)
train_short <- training(splits) %>% slice_sample(prop = 0.5, by = symbol)

# * remove unneeded data from environment ----
rm(prices_base)
rm(prices_dt)
rm(prices_features_list)
rm(data_prepared_dt)
gc()

# 3.0 RECIPE ----

# * Recipe Specification ----

recipe_spec <- recipe(Return_fwd_21 ~ ., data = training(splits)) %>%
    update_role(rowid, new_role = 'identifier') %>% 
    # update_role(symbol, new_role = 'symbol') %>%
    # step_mutate_at(symbol, fn = droplevels) %>%
    # step_timeseries_signature(date) %>%
    # step_rm(matches("(.xts$)|(.iso$)|(.hour)|(.minute)|(.second)|(.am.pm)")) %>%
    # step_rm(date_index.num, date_year) %>%
    # step_normalize(date_index.num, ends_with("_year")) %>%  
    # step_normalize(index.num,year) %>% 
    # step_rm(Return_fwd_5, Return_fwd_10) %>%
    step_dummy(all_nominal_predictors(), one_hot = T, keep_original_cols = F) %>%
    # step_mutate(spy_state = as.factor(spy_state)) %>% 
    step_interact(~Close_macd_long:Close_macd_short) %>% 
    step_interact(~Close_macd_long_signal:Close_macd_short_signal) %>% 
    step_filter_missing(all_predictors(), threshold = 0.2) %>% 
    step_zv(all_predictors()) #%>% 
    # step_nzv(all_predictors(), unique_cut = 0.02)

# 4.0 HYPERPARAMETER TUNING MODELS ---- 

# * RESAMPLES - K-FOLD ----- 
set.seed(69)
# resamples_kfold <- training(splits) %>% vfold_cv(v = 5)
resamples_kfold_short <- train_short %>% vfold_cv(v = 5)

# * Parallel Processing ----
# cl <- parallel::makeCluster(2, timeout = 60)
# plan(cluster, workers = cl)

options(future.globals.maxSize = 1.3 * 1024^3)  # Set future size to fit memory 2.0 GiB

parallel_start(1:6, .method = "future")

parallel_stop()

plan(sequential)

# * LightGBM TUNE ----

# ** Tunable Specification ----

model_spec_lgb_tune <- boost_tree(
  "regression",
  mtry           = tune(),
  trees          = tune(),
  min_n          = tune(),
  tree_depth     = tune(),
  learn_rate     = tune(),
  #loss_reduction = 0.01,#tune()
  stop_iter      = 20) %>% 
  set_engine('lightgbm', counts = F, validation = 0.2, num_threads = -1)

wflw_spec_lgb_tune <- workflow() %>% 
    add_model(model_spec_lgb_tune) %>% 
    add_recipe(recipe_spec %>% step_rm(date))

# ** Tuning
set.seed(69)
start <- Sys.time()
tune_results_lgb <- wflw_spec_lgb_tune %>% 
  tune_race_anova(
    # resamples = resamples_kfold,
    resamples = resamples_kfold_short,
    param_info = extract_parameter_set_dials(wflw_spec_lgb_tune) %>% 
      update(learn_rate = learn_rate(range = c(0.05, 0.5), trans = NULL),
             trees      = trees(range = c(200,3500)),
             mtry       = mtry_prop(range = c(0.05,0.8))
      ),
    grid = 10,
    control = control_race(verbose = T, parallel_over = NULL)
  )
end <- Sys.time()
end-start

# ** Results

tune_results_lgb %>% 
  show_best(metric = "rmse", n = Inf)

tune_results_lgb %>% 
  show_best(metric = "rsq", n = Inf)

# ** Finalize

wflw_fit_lgb_tuned <- wflw_spec_lgb_tune %>% 
  finalize_workflow(select_best(tune_results_lgb, metric = "rmse")) %>% 
  fit(training(splits))

# ** testing accuracy ----
augment(wflw_fit_lgb_tuned,testing(splits)) %>% 
    mutate(error = .pred-Return_fwd_21) %>% 
    summarise(rmse = sqrt(mean(error^2)))

augment(wflw_fit_lgb_tuned,testing(splits)) %>% 
    rsq(.pred, Return_fwd_21)

fcst_test_fit_lgb_tuned <- modeltime_table(wflw_fit_lgb_tuned) %>% 
    modeltime_calibrate(new_data = testing(splits), id = "symbol") %>% 
    modeltime_forecast(actual_data = data_prepared_dt_filter, 
                       new_data = testing(splits),
                       keep_data = T)

lgboost_directions <- fcst_test_fit_lgb_tuned %>%
    filter(.key != "actual") %>% 
    select(.index, symbol, .value, Return_fwd_21) %>% 
    rename(prediction = .value, actual = Return_fwd_21) %>% 
    mutate(direction = if_else((actual >= 0 & prediction >= 0)|
                                   (actual < 0 & prediction < 0),
                               1, 0),
           model = "light_gbm") 

lgboost_directions %>% 
    summarise(wins = sum(direction), rate = mean(direction), count = n())

fcst_test_fit_lgb_tuned %>% 
    filter(symbol == testing_symbol) %>% 
  plot_modeltime_forecast(.conf_interval_show = F)

# ** save tune results ----
write_rds(tune_results_lgb, "02_models/tune_results_lgb.rds")
rm(tune_results_lgb)
rm(wflw_spec_lgb_tune) # can recreate spec easily

gc()

# * XGBOOST TUNE ----

# ** Tunable Specification ----
model_spec_xgboost_tune <- boost_tree(
    "regression",
    mtry           = tune(),
    trees          = tune(),
    min_n          = tune(),
    tree_depth     = tune(),
    learn_rate     = tune(),
    # loss_reduction = 0.0005,#tune(),
    stop_iter      = 20
    ) %>% 
    set_engine('xgboost', 
               counts = F, 
               nthread =  -1, 
               tree_method = "hist",
               validation = 0.2)

wflw_spec_xgboost_tune <- workflow() %>% 
    add_model(model_spec_xgboost_tune) %>% 
    add_recipe(recipe_spec %>% step_rm(date)) 

# ** Tuning

set.seed(69)
start <- Sys.time()
tune_results_xgboost <- wflw_spec_xgboost_tune %>% 
    tune_race_anova(
        # resamples = resamples_kfold,
        resamples = resamples_kfold_short,
        param_info = extract_parameter_set_dials(wflw_spec_xgboost_tune) %>% 
          update(learn_rate = learn_rate(range = c(0.05, 0.5), trans = NULL),
                 trees      = trees(range = c(200,3500)),
                 mtry       = mtry_prop(range = c(0.05,0.8))
                 ),
        grid = 10,
        control = control_race(verbose = T, parallel_over = NULL)
    )
end <- Sys.time()
end-start

# ** Results

tune_results_xgboost %>% 
    show_best(metric = "rmse", n = Inf)

tune_results_xgboost %>% 
    show_best(metric = "rsq", n = Inf)


# ** Finalize

start <- Sys.time()
wflw_fit_xgboost_tuned <- wflw_spec_xgboost_tune %>% 
    finalize_workflow(select_best(tune_results_xgboost, metric = "rmse")) %>% 
    fit(training(splits))
end <- Sys.time() 
end-start

# ** test accuracy ----
augment(wflw_fit_xgboost_tuned,testing(splits)) %>% 
    mutate(error = .pred-Return_fwd_21) %>% 
    summarise(rmse = sqrt(mean(error^2)))

augment(wflw_fit_xgboost_tuned,testing(splits)) %>% 
    rsq(.pred, Return_fwd_21)

fcst_test_fit_xgboost_tuned <- modeltime_table(wflw_fit_xgboost_tuned) %>% 
    modeltime_calibrate(new_data = testing(splits), id = "symbol") %>% 
    modeltime_forecast(actual_data = data_prepared_dt_filter, 
                       new_data = testing(splits),
                       keep_data = T)

xgboost_directions <- fcst_test_fit_xgboost_tuned %>%
    filter(.key != "actual") %>% 
    select(.index, symbol, .value, Return_fwd_21) %>% 
    rename(prediction = .value, actual = Return_fwd_21) %>% 
    mutate(direction = if_else((actual >= 0 & prediction >= 0)|
                                   (actual < 0 & prediction < 0),
                               1, 0),
           model = "xg_boost") 

xgboost_directions %>% 
    summarise(wins = sum(direction), rate = mean(direction), count = n())

fcst_test_fit_xgboost_tuned  %>% 
    filter(symbol == testing_symbol) %>% 
    plot_modeltime_forecast(.conf_interval_show = F)

#calibrate_and_plot(wflw_fit_xgboost_tuned, plot = F)

# ** save tune results ----
write_rds(tune_results_xgboost, "02_models/tune_results_xgboost.rds", compress = "gz")
rm(tune_results_xgboost)
rm(wflw_spec_xgboost_tune) # remove spec to save memory

gc()

# * Prophet XGBoost TUNE ----

# ** Tunable Specification ----

model_spec_prophet_boost_tune <- prophet_boost(
    #changepoint_num    = 25,
    #changepoint_range  = 0.8,
    seasonality_yearly = F, #all seasonalities to F because xgboost does that
    seasonality_weekly = F,
    seasonality_daily  = F,
    mtry              = tune(),
    trees             = 400, # leave so it stabilizes the model
    #min_n             = tune(),
    tree_depth        = tune(),
    learn_rate        = tune(),
    #loss_reduction    = 0.0005, #tune()
    stop_iter         = 20
) %>% set_engine("prophet_xgboost",  
                 counts = F, 
                 nthread =  -1, 
                 tree_method = "hist"
                 ,validation = 0.2)

wflw_spec_prophet_boost_tune <- workflow() %>% 
    add_model(model_spec_prophet_boost_tune) %>% 
    add_recipe(recipe_spec)

# ** Tuning

set.seed(69)
start <- Sys.time()
tune_results_prophet_boost <- wflw_spec_prophet_boost_tune %>% 
    tune_race_anova(
        # resamples = resamples_kfold,
        resamples = resamples_kfold_short,
        param_info = extract_parameter_set_dials(wflw_spec_prophet_boost_tune) %>% 
            update(learn_rate = learn_rate(range = c(0.1, 0.4), trans = NULL),
                   #trees      = trees(range = c(50,1000)),
                   mtry       = mtry_prop(range = c(0.1,0.8))
            ),
        grid = 5,
        control = control_race(verbose_elim = T, parallel_over = NULL)
    )
end <- Sys.time() 
end-start

# ** Results

tune_results_prophet_boost %>% 
    show_best(metric = "rmse", n = Inf)

tune_results_prophet_boost %>% 
    show_best(metric = "rsq", n = Inf)

# ** Finalize

start <- Sys.time()
wflw_fit_prophet_boost_tuned <- wflw_spec_prophet_boost_tune %>% 
    finalize_workflow(select_best(tune_results_prophet_boost, metric = "rmse")) %>% 
    fit(training(splits))
end <- Sys.time()
end - start

# ** testing accuracy ----
augment(wflw_fit_prophet_boost_tuned,testing(splits)) %>% 
    mutate(error = .pred-Return_fwd_21) %>% 
    summarise(rmse = sqrt(mean(error^2)))

augment(wflw_fit_prophet_boost_tuned,testing(splits)) %>% 
    rsq(.pred, Return_fwd_21)

fcst_test_fit_prophet_boost_tuned <- modeltime_table(wflw_fit_prophet_boost_tuned) %>% 
    modeltime_calibrate(new_data = testing(splits), id = "symbol") %>% 
    modeltime_forecast(actual_data = data_prepared_dt_filter, 
                       new_data = testing(splits),
                       keep_data = T)

prophet_boost_directions <- fcst_test_fit_prophet_boost_tuned %>%
    filter(.key != "actual") %>% 
    select(.index, symbol, .value, Return_fwd_21) %>% 
    rename(prediction = .value, actual = Return_fwd_21) %>% 
    mutate(direction = if_else((actual >= 0 & prediction >= 0)|
                                   (actual < 0 & prediction < 0),
                               1, 0),
           model = "prophet_boost") 

prophet_boost_directions %>% 
    summarise(wins = sum(direction), rate = mean(direction), count = n())

fcst_test_fit_prophet_boost_tuned %>% 
    filter(symbol == testing_symbol) %>% 
    plot_modeltime_forecast(.conf_interval_show = F)

# calibrate_and_plot(wflw_fit_prophet_boost_tuned, plot = F)

# ** save tune results ----
write_rds(tune_results_prophet_boost, "02_models/tune_results_prophet_boost.rds")
rm(tune_results_prophet_boost)
rm(wflw_spec_prophet_boost_tune) # remove spec to save memory

gc()

# * glmnet TUNE ----

# ** Tunable Specification ----

model_spec_glmnet_tune <- linear_reg(
  mode = "regression",
  penalty = tune(), # higher penalty removes features faster
  mixture = tune()) %>% 
  set_engine('glmnet')

wflw_spec_glmnet_tune <- workflow() %>% 
  add_model(model_spec_glmnet_tune) %>% 
  add_recipe(recipe_spec %>% 
                 step_rm(date) %>% 
                 step_ts_impute(all_numeric_predictors()))

# ** Tuning
set.seed(69)
start <- Sys.time()
tune_results_glmnet <- wflw_spec_glmnet_tune %>% 
  tune_race_anova(
    resamples = resamples_kfold_short,
    # resamples = resamples_kfold,
    grid = 6,
    control = control_race(verbose = T, parallel_over = NULL)
  )
end <- Sys.time()
end-start

# ** Results
tune_results_glmnet %>% 
  show_best(metric = "rmse", n = Inf)

tune_results_glmnet %>% 
    show_best(metric = "rsq", n = Inf)

# ** Finalize

wflw_fit_glmnet_tuned <- wflw_spec_glmnet_tune %>% 
  finalize_workflow(select_best(tune_results_glmnet, metric = "rmse")) %>% 
  fit(training(splits))

# ** test accuracy ----
augment(wflw_fit_glmnet_tuned,testing(splits)) %>% 
    mutate(error = .pred-Return_fwd_21) %>% 
    summarise(rmse = sqrt(mean(error^2)))

augment(wflw_fit_glmnet_tuned,testing(splits)) %>% 
    rsq(.pred, Return_fwd_21)

fcst_test_fit_glmnet_tuned <- modeltime_table(wflw_fit_glmnet_tuned) %>% 
    modeltime_calibrate(new_data = testing(splits), id = "symbol") %>% 
    modeltime_forecast(actual_data = data_prepared_dt_filter, 
                       new_data = testing(splits),
                       keep_data = T)

glmnet_directions <- fcst_test_fit_glmnet_tuned %>%
    filter(.key != "actual") %>% 
    select(.index, symbol, .value, Return_fwd_21) %>% 
    rename(prediction = .value, actual = Return_fwd_21) %>% 
    mutate(direction = if_else((actual >= 0 & prediction >= 0)|
                                   (actual < 0 & prediction < 0),
                               1, 0),
           model = "glmnet") 

glmnet_directions %>% 
    summarise(wins = sum(direction), rate = mean(direction), count = n())

fcst_test_fit_glmnet_tuned %>% 
    filter(symbol == testing_symbol) %>% 
    plot_modeltime_forecast(.conf_interval_show = F)

# calibrate_and_plot(wflw_fit_glmnet_tuned, plot =F)

# save tune results
write_rds(tune_results_glmnet, "02_models/tune_results_glmnet.rds")
rm(tune_results_glmnet)
rm(wflw_spec_glmnet_tune) # remove spec to save memory

gc()

# * CatBoost ----
# https://catboost.ai/docs/en/
# https://bonsai.tidymodels.org/
# https://bonsai.tidymodels.org/reference/train_catboost.html

model_spec_catboost <- boost_tree("regression",
                                  trees  = 2500
                                  #,tree_depth = 5
                                  #,min_n = 20
                                  #,mtry = 5
                                ) %>% 
    set_engine('catboost',
               early_stopping_rounds = 30, 
               boosting_type = "Plain") 

set.seed(69)
start <- Sys.time()
wflw_fit_catboost <- workflow() %>% 
    add_model(model_spec_catboost) %>% 
    add_recipe(recipe_spec %>% step_rm(date)) %>% 
    fit(training(splits))
    #fit_resamples(resamples_kfold)
end <- Sys.time()
end-start

wflw_fit_catboost
# collect_metrics(wflw_fit_cb)

extract_fit_engine(wflw_fit_catboost) %>% 
    catboost::catboost.get_feature_importance(model = .) %>%
    as_tibble(rownames = 'Variable') %>% 
    #enframe() %>%
    arrange(desc(V1))

# ** accuracy on testing ----

augment(wflw_fit_catboost,training(splits)) %>% 
    mutate(error = .pred-Return_fwd_21) %>% 
    summarise(rmse = sqrt(mean(error^2)))

augment(wflw_fit_catboost,training(splits)) %>% 
    rsq(.pred, Return_fwd_21)

augment(wflw_fit_catboost,testing(splits)) %>% 
    mutate(error = .pred-Return_fwd_21) %>% 
    summarise(rmse = sqrt(mean(error^2)))

augment(wflw_fit_catboost,testing(splits)) %>% 
    rsq(.pred, Return_fwd_21)

# calibrate_and_plot(wflw_fit_cb, plot = F)

fcst_test_fit_catboost_tuned <- modeltime_table(wflw_fit_catboost) %>% 
    modeltime_calibrate(new_data = testing(splits), id = "symbol") %>% 
    modeltime_forecast(actual_data = data_prepared_dt_filter, 
                       new_data = testing(splits),
                       keep_data = T)

catboost_directions <- fcst_test_fit_catboost_tuned %>%
    filter(.key != "actual") %>% 
    select(.index, symbol, .value, Return_fwd_21) %>% 
    rename(prediction = .value, actual = Return_fwd_21) %>% 
    mutate(direction = if_else((actual >= 0 & prediction >= 0)|
                                   (actual < 0 & prediction < 0),
                               1, 0),
           model = "catboost") 

catboost_directions %>% 
    # group_by(symbol) %>% 
    summarise(wins = sum(direction), rate = mean(direction), count = n()) %>% 
    arrange(desc(rate))

fcst_test_fit_catboost_tuned %>% 
    filter(symbol == testing_symbol) %>% 
    plot_modeltime_forecast(.conf_interval_show = F)

# * save tune results SKIP until we have tuning on catboost ----
write_rds(tune_results_catboost, "02_models/tune_results_catboost.rds")
rm(tune_results_catboost)
rm(wflw_spec_catboost_tune) # can recreate spec easily

gc()

# * Modeling explanation ----

extract_fit_engine(wflw_fit_xgboost_tuned) %>%
  xgboost::xgb.importance(model = .) %>%
  as_tibble() %>% View()
  slice_max(Gain, n = 20) %>% 
  mutate(Feature = as_factor(Feature) %>% fct_rev()) %>% 
  ggplot(aes(x = Gain, y = Feature, fill = Feature))+
  geom_col(alpha = 0.7)+
  scale_fill_tq()+
  theme_few()+
  theme(legend.position = "none")+
  ggtitle("XGBoost Feature Importance")

extract_fit_engine(wflw_fit_lgb_tuned) %>%
  # extract_fit_engine(wflw_fit_lightgbm) %>%
  lightgbm::lgb.importance(model = .) %>% 
  as_tibble() %>% View()
  slice_max(order_by = Gain, n = 20) %>% 
  mutate(Feature = as_factor(Feature) %>% fct_rev()) %>% 
  ggplot(aes(x = Gain, y = Feature, fill = Feature))+
  geom_col(alpha = 0.7)+
  scale_fill_tq()+
  theme_few()+
  theme(legend.position = "none")+
  ggtitle("LightGBM Feature Importance")

extract_fit_engine(wflw_fit_xgboost_tuned) %>%
extract_fit_engine(wflw_fit_lgb_tuned) %>%
    lightgbm::lgb.importance(model = .) %>% 
    as_tibble() %>% 
  arrange(desc(Gain)) %>% 
  mutate(rank = row_number()) %>% View()
  filter(str_detect(Feature, "date"))

# ** compare xgb and lgbm ----
library(tidytext)
bind_rows(extract_fit_engine(wflw_fit_xgboost_tuned) %>%
              xgboost::xgb.importance(model = .) %>%
              as_tibble() %>% 
              filter(!str_detect(Feature,"symbol")) %>% 
              arrange(desc(Gain)) %>% 
              mutate(rank = row_number(),
                     model = "xgboost"),
          extract_fit_engine(wflw_fit_lgb_tuned) %>%
              # extract_fit_engine(wflw_fit_lightgbm) %>%
              lightgbm::lgb.importance(model = .) %>% 
              as_tibble() %>% 
              filter(!str_detect(Feature,"symbol")) %>%
              arrange(desc(Gain)) %>% 
              mutate(rank = row_number(),
                     model = "lightgbm")) %>% View()
    slice_max(Gain, n = 20) %>% 
    ggplot(aes(x = Gain,
               y = reorder_within(x = Feature, by = Gain, within = model),
               fill = model)
           )+
    geom_col(show.legend = F)+
    facet_wrap(.~model, scales = 'free_y')+
    scale_y_reordered()+
    theme_bw()+
    ylab("")

multi_calibrate <- modeltime_table(
    wflw_fit_lgb_tuned,
    wflw_fit_xgboost_tuned,
    wflw_fit_prophet_boost_tuned
    ,wflw_fit_cb
    ) %>% 
    modeltime_calibrate(new_data = testing(splits) %>% 
                            filter(symbol == testing_symbol)) 

multi_calibrate %>% 
    modeltime_forecast(actual_data = data_prepared_dt_filter %>% 
                           filter(symbol == testing_symbol), 
                       new_data = testing(splits) %>% 
                           filter(symbol == testing_symbol)) %>% 
    plot_modeltime_forecast(.conf_interval_show = F)

multi_calibrate %>%
    modeltime_accuracy()

# scale_fill_viridis_d()

# ** SHAP Analysis ----
fit <- extract_fit_parsnip(wflw_fit_lgb_tuned)

df_explain  <- bake( 
  prep(recipe_spec), 
  has_role("predictor"),
  new_data = training(splits) %>% slice_sample(prop = 0.1)
  ) %>% 
  select(-date) %>% 
    select(all_of(fit[["preproc"]][["x_names"]]))

shap_values <- extract_fit_engine(wflw_fit_lgb_tuned) %>% 
  shapviz(X_pred = data.matrix(df_explain %>% select(-date)), 
          X = df_explain)

# SHAP importance
shap_values %>% 
  sv_importance(show_numbers = TRUE, "beeswarm", alpha = 0.7) +
  ggtitle("SHAP importance")+
  scale_color_gradient(high = "#ff0d57", low = "#1e88e5")+
  theme_classic()

# python shap colors 
#c("#1E88E5", "#ff0d57", "#13B755", "#7C52FF", "#FFC000", "#00AEEF")

shap_values %>% 
  sv_importance(show_numbers = TRUE, fill = "#1e88e5", alpha = 0.7) +
  ggtitle("SHAP importance")+
  theme_classic()

shap_names <- shap_values %>% 
  sv_importance("no") %>% enframe() %>% arrange(desc(value)) %>% 
    mutate(rank = row_number())

shap_values %>% 
  sv_dependence(shap_names$name[1], alpha = 0.7) +
  scale_color_gradient(high = "#ff0d57", low = "#1e88e5")+
  theme_minimal()

shap_values %>% 
  sv_dependence(shap_names$name[2], alpha = 0.7) +
  scale_color_gradient(high = "#ff0d57", low = "#1e88e5")+
  theme_minimal()

shap_values %>% 
  sv_dependence(shap_names$name[7], alpha = 0.7) +
  scale_color_gradient(high = "#ff0d57", low = "#1e88e5")+
  geom_smooth(se = F,method = "gam",colour = "gray",linewidth = 1)+
  theme_minimal()

shap_values %>% 
  sv_dependence("Close_macd_long_signal_trend", alpha = 0.7) +
  scale_color_gradient(high = "#ff0d57", low = "#1e88e5")+
  geom_smooth(se = F,colour = "gray",linewidth = 1)+
  theme_minimal()

shap_values %>% 
  sv_dependence("Close_macd_short_signal_trend", alpha = 0.7) +
  scale_color_gradient(high = "#ff0d57", low = "#1e88e5")+
  geom_smooth(se = F,colour = "gray",linewidth = 1)+
  theme_minimal()

shap_values %>% 
  sv_dependence("Close_momentum_21_252_126", alpha = 0.7) +
  scale_color_gradient(high = "#ff0d57", low = "#1e88e5")+
  theme_minimal()

xvars <- c("Vol_WAP_norm","Close_momentum_21_252_126","Close_natr_63","Close_rel_vol_14")
evars <- colnames(shap_values$S)[grep("eps", colnames(shap_values$S))]
dvars <- colnames(shap_values$S)[grep("div", colnames(shap_values$S))]
mvars <- colnames(shap_values$S %>% as_tibble() %>% select(contains("macd")))
vvars <- colnames(shap_values$S)[grep("WAP", colnames(shap_values$S))]
avars <- colnames(shap_values$S)[grep("ADX", colnames(shap_values$S))]

shap_values %>% 
  sv_dependence(vvars, viridis_args = list(option = "viridis", direction = -1))

shap_values %>% 
  sv_dependence(evars, viridis_args = list(option = "viridis", direction = -1))

shap_values %>% 
  sv_dependence(dvars, viridis_args = list(option = "viridis", direction = -1))

shap_values %>% 
    sv_dependence(mvars, viridis_args = list(option = "viridis", direction = -1))

shap_values %>% 
    sv_dependence(avars, viridis_args = list(option = "viridis", direction = -1))


# SHAP interactions for fwd return
shap_values %>%  
  sv_dependence("Vol_WAP_norm", 
                color_var = xvars, 
                interactions = TRUE,  
                viridis_args = list(option = "viridis", direction = -1))

shap_values %>% 
  sv_dependence2D("Vol_WAP_norm","Return_fwd_21", alpha = 0.7) +
  scale_color_gradient(high = "#ff0d57", low = "#1e88e5")+
  theme_minimal()


# 5.0 EVALUATE TUNED FORECASTS  -----
# * Model Table ----
submodels_tbl <- modeltime_table(
    wflw_fit_xgboost_tuned
    ,wflw_fit_lgb_tuned
    ,wflw_fit_prophet_boost_tuned
    ,wflw_fit_catboost
    ,wflw_fit_glmnet_tuned
) %>% 
    update_model_description(1, "XGBOOST - Tuned") %>% 
    update_model_description(2, "LightGBM - Tuned") %>% 
    update_model_description(3, "Prophet Boost - Tuned") %>% 
    update_model_description(4, "CatBoost") %>% 
    update_model_description(5, "GLMNet - Tuned")

# * Calibration ----
calibration_tbl <- submodels_tbl %>% 
    modeltime_calibrate(testing(splits), id = "symbol")

# * Accuracy ----
calibration_tbl %>% 
    modeltime_accuracy(metric_set = extended_forecast_accuracy_metric_set()) %>% 
    arrange(rmse)

# * Accuracy summary plot ----
calibration_tbl %>% 
  modeltime_accuracy(acc_by_id = T) %>% #View()
  #arrange(rmse) %>% 
  select(.model_desc, rmse:rsq) %>% #group_by(.model_desc) %>% get_summary_stats()
  pivot_longer(-.model_desc) %>% 
  ggplot(aes(x = .model_desc, y = value, fill = .model_desc))+
  geom_boxplot(alpha = 0.7, show.legend = F) +
  facet_grid(name~., scales = "free_y")+
    scale_fill_tq()+
  theme_tq()

# * Accuracy by symbol ----
acc_by_symbol <- calibration_tbl %>% 
    modeltime_accuracy(acc_by_id = TRUE) %>% 
    select(-.type) %>% 
    pivot_longer(-(.model_id:symbol)) %>% 
    group_by(symbol, name) %>%
    summarise(value = mean(value)) %>% 
    ungroup() %>% 
    pivot_wider(names_from = name, values_from = value)

write_rds(acc_by_symbol, str_glue("02_models/{today()}_acc_by_symbol.rds"))

# * Forecast symbols ----
# extract the subset of symbols that have the lowest rmse
# can do either a fixed n or a percentage

forecast_symbols <- acc_by_symbol %>% 
    slice_min(rmse, n = 100) %>%
    #slice_min(rmse, prop = 0.4) %>%
    droplevels() %>% 
    arrange(symbol) %>% 
    pull(symbol)

# * Forecast Test ----
gc()
forecast_test <- calibration_tbl %>% 
    modeltime_forecast(
        new_data   = testing(splits),
        actual_data = data_prepared_dt_filter,
        keep_data  = T, # keeps the grouping variable and base data
        conf_by_id = T) 



# * backtesting ----
back_fcst <- calibration_tbl %>% 
    modeltime_forecast(
        new_data = data_prepared_dt_filter %>% 
            filter(date %between% c("2024-12-01","2024-12-10")) %>% 
            drop_na(),
        actual_data = data_prepared_dt_filter %>% 
            filter(date %between% c("2024-11-01","2024-12-10")) %>% 
            drop_na(),
        keep_data = T,
        conf_by_id = T
    ) 

back_fcst %>% 
    filter(date == "2024-12-06" & .model_desc != "ACTUAL") %>% 
    arrange(desc(.value))

backtest <- back_fcst %>% 
    filter(date >= "2024-12-01") %>% 
    select(symbol, .value, .model_desc, date) %>% 
    pivot_wider(names_from = .model_desc, values_from = .value, 
                names_repair = "universal") %>%
    group_by(symbol, date) %>% 
    mutate(forecast_mean = mean(c_across(c(contains("Tuned"))), na.rm=TRUE),
           forecast_median = median(c_across(c(contains("Tuned"))), na.rm=TRUE)) %>% 
    ungroup() %>% 
    mutate(correct_mean = if_else((forecast_median < 0 & ACTUAL < 0)|
                                      (forecast_median >= 0 & ACTUAL >= 0),1,0),
           correct_median = if_else((forecast_median < 0 & ACTUAL < 0)|
                                        (forecast_median >= 0 & ACTUAL >= 0),1,0)) 

#symbol %in% c("NVDA","GE","FI","FIS","CTAS","WELL","MSFT","MMM","IRM","NVDA","TT")

backtest %>% View()
rmse(backtest, truth = ACTUAL, estimate = forecast_median)
rmse(backtest, truth = ACTUAL, estimate = forecast_mean)

back_fcst %>% 
    plot_modeltime_forecast(.facet_vars = symbol, .facet_ncol = 3)

# ** forecast test plot ----
forecast_test %>% 
    group_by(symbol) %>%
    #filter(symbol == "AAP") %>% 
    # filter(symbol %in% c("NVDA","GE","FI","FIS","CTAS","WELL","MSFT","MMM","IRM")) %>% 
    filter(.index >= "2025-08-01") %>% 
    plot_modeltime_forecast(
        .facet_ncol = 2,
        .conf_interval_show = F,
        .interactive = T,
        .trelliscope = T
    )

test_fcst <- forecast_test %>% 
    filter(date == max(date)) %>% 
    slice_sample(n = 50) %>% 
    select(.model_id, .model_desc, symbol,.key, date, .value)

test_fcst %>% 
    group_by(symbol, .key) %>% 
    summarise(mean = mean(.value), median = median(.value)) %>% 
    pivot_wider(names_from = .key, values_from = mean:median,names_sort = T)


# 6.0 RESAMPLING ----
# - Assess the stability of our models over time
# - Helps us strategize an ensemble approach

# * Time Series CV ----

resamples_tscv <- training(splits) %>% 
    time_series_cv(
        initial     = 252,
        assess      = 21,
        skip        = 21,
        cumulative  = T,
        slice_limit = 12
    )

resamples_tscv %>% 
    tk_time_series_cv_plan() %>% 
    plot_time_series_cv_plan(date, Return_fwd_21,
                             .facet_ncol = 3,
                             .interactive = F)
  

# * Fitting Resamples ----

model_tbl_tuned_resamples <- submodels_tbl %>% 
  #modeltime_table(wflw_fit_lgb_tuned) %>% 
    modeltime_fit_resamples(
        resamples = resamples_tscv,
        control   = control_resamples(verbose = T, parallel_over = NULL)
    )

# * Resampling Accuracy Table ----

model_tbl_tuned_resamples %>% 
    modeltime_resample_accuracy(
        metric_set = metric_set(mae, rmse, rsq),
        summary_fns = list(mean = mean, sd = sd)
    ) %>% 
    arrange(rmse_mean)

model_tbl_tuned_resamples %>% 
  modeltime_resample_accuracy(
    metric_set = metric_set(rmse, rsq),
    summary_fns = NULL
  ) 

# * Resampling Accuracy Plot ----
model_tbl_tuned_resamples %>% 
    plot_modeltime_resamples(
        .metric_set = metric_set(mae, rmse, rsq),
        .point_size = 2,
        .point_alpha = 0.8,
        .facet_ncol = 1
    )

# 7.0 ENSEMBLE PANEL MODELS -----

# * Average Ensemble ----
# submodels_ids_to_keep <- c(1,3,4)

ensemble_fit <- calibration_tbl %>%
    ensemble_average(type = 'median') # reduces effects of bad forecasts and reduces overfitting. probably not needed with only 3 forecasts

model_ensemble_tbl <- modeltime_table(ensemble_fit)

# * Accuracy ----
model_ensemble_tbl %>% 
  modeltime_accuracy(testing(splits),
                     metric_set = extended_forecast_accuracy_metric_set())

#model_ensemble_tbl %>% 
    
# * Weighted ensemble ----
loadings_tbl <- submodels_tbl %>% 
  #filter(.model_id %in% submodels_ids_to_keep) %>% 
  modeltime_accuracy(testing(splits)) %>%
  mutate(rank = min_rank(-rmse))

ensemble_fit_wt <- submodels_tbl %>%
  ensemble_weighted(loadings = loadings_tbl$rank)

ensemble_fit_wt$fit$loadings_tbl

model_ensemble_tbl_wt <- modeltime_table(ensemble_fit_wt)


# * Calibrate ensemble ----

model_ensemble_tbl_wt %>% 
    modeltime_calibrate(testing(splits),id = "symbol") %>%
    modeltime_accuracy(testing(splits), 
                     metric_set = extended_forecast_accuracy_metric_set())

# write_rds(model_ensemble_tbl, 
#           str_glue("02_models/{today()}_model_ensemble_tbl.rds"),
#           compress = "gz")

# 8.0 Ensemble Forecast test ----
gc()
forecast_ensemble_test_tbl <- model_ensemble_tbl_wt %>% 
    modeltime_forecast(
        new_data = testing(splits),
        actual_data = data_prepared_dt_filter,
        keep_data = T
    )

ensemble_directions <- forecast_ensemble_test_tbl %>%
    filter(.key != "actual") %>% 
    select(.index, symbol, .value, Return_fwd_21) %>% 
    rename(prediction = .value, actual = Return_fwd_21) %>% 
    mutate(direction = if_else((actual >= 0 & prediction >= 0)|
                                   (actual < 0 & prediction < 0),
                               1, 0),
           model = "xg_boost") 

ensemble_directions %>% 
    summarise(wins = sum(direction), rate = mean(direction), count = n())

directions <- rbind(ls(pattern = "direction"))

# * ensemble forecast test plot ----
forecast_ensemble_test_tbl %>% filter(symbol == testing_symbol) %>% plot_modeltime_forecast(.conf_interval_show = FALSE)
    group_by(symbol) %>% 
    plot_modeltime_forecast(
        .facet_ncol = 3,
        #.conf_interval_show = F,
        .interactive = T,
        .legend_show = F,
        .trelliscope = T
    )

# * ensemble forecast test table ----
forecast_ensemble_test_tbl %>% 
    filter(.key == 'prediction') %>% 
    select(symbol, .value, Return_fwd_21) %>% 
    #group_by(symbol) %>% 
    summarize_accuracy_metrics(
        truth = Return_fwd_21,
        estimate = .value,
        metric_set = metric_set(mae, rmse, rsq)
    )

# 9.0 Final Forecasting ----
# * Refit ----
data_prepared_clean_dt <- data_prepared_dt_filter %>% 
    filter_by_time(date, .start = max(date)-years(2)) %>% 
    filter(symbol %in% forecast_symbols) %>% 
    #droplevels() %>% 
    drop_na()

model_ensemble_refit_tbl <- model_ensemble_tbl_wt %>% 
    modeltime_refit(data_prepared_clean_dt)

# * Ensemble final forecast ----
final_forecast_dt <- forecast_dt[symbol %in% forecast_symbols]

model_ensemble_final_forecast <- model_ensemble_refit_tbl %>% 
    modeltime_forecast(
        new_data    = final_forecast_dt,
        actual_data = data_prepared_clean_dt,
        keep_data   = T,
        conf_by_id  = T
    )  

# * plot final forecast ----
model_ensemble_final_forecast %>% 
  arrange(desc(date)) %>% 
    group_by(symbol) %>% 
    plot_modeltime_forecast(
        .facet_vars = symbol,
        .facet_ncol = 3,
        .y_intercept = 0,
        .conf_interval_show = F,
        .legend_show = F,
        .trelliscope = F
    )

# * View final forecasts ----
model_ensemble_final_forecast %>% 
    filter(date == max(date)) %>% 
    select(symbol, .value, date) %>% 
    arrange(desc(.value)) %>% 
    left_join(acc_by_symbol) %>% View()

model_ensemble_final_forecast %>% 
    filter(date == max(date)) %>% 
    select(symbol, .value, date) %>% 
    arrange(desc(.value)) %>% 
    left_join(acc_by_symbol) %>%
    filter(.value >= 0) %>% 
    # mutate(value = rsq/rmse) %>% 
    slice_min(rmse, n = 30) %>% 
    #filter(rmse < 0.07) %>% 
    #slice_max(value, n = 30) %>% 
    arrange(desc(.value))

# * Turn OFF Parallel Backend ----
plan(sequential)
parallel_stop()

# 10.0 save forecasts ----
write_rds(model_ensemble_final_forecast,
          str_glue("01_save_data/01_saved_forecasts/{today()}_model_ensemble_final_forecast.rds"))
