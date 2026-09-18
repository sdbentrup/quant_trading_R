options(scipen = 999)

get <- forecast_acc_sybmol %>% 
    select(symbol, date, .value, rmse, rsq, ev) %>% 
    # filter(rmse < 0.05) %>% 
    # filter(.value > 0) %>% 
    # slice_min(rmse, n = 50) %>% 
    # slice_max(.value, n = 10) %>% 
    # mutate(ev = (1-rmse) * .value) %>% 
    slice_max(ev, n = 10) %>% 
    pull(symbol)

train_date <- today() - years(3)

s <- tq_get(get, from = train_date-years(1))
s <- tq_get(cb_fcst_get, from = train_date-years(1))
s %>% plot_ly(x = ~date, y = ~close, color = ~symbol,mode = 'lines')

s %>%
    ggplot(aes(x = date, y = close)) +
    geom_candlestick(aes(open = open, high = high, low = low, close = close),
                     colour_up = "darkgreen", colour_down = "darkred", 
                     fill_up  = "darkgreen", fill_down  = "darkred") +
    labs(y = "Closing Price", x = "") + 
    facet_wrap(~ symbol, scale = "free_y") +
    theme_tq()

returns <- s %>%
    group_by(symbol) %>%
    tq_transmute(select = close,
                 mutate_fun = periodReturn,
                 period = 'monthly',
                 col_rename = "close_ret") %>% 
    ungroup()

returns %>% plot_ly(x = ~date, y = ~close_ret, color = ~symbol,mode = 'lines')

returns %>% 
    tq_portfolio(assets_col   = symbol,
                 returns_col  = close_ret,
                 #weights      = wts,
                 col_rename   = "investment.growth",
                 wealth.index = TRUE) %>%
    mutate(investment.growth = investment.growth * 10000) %>% 
    plot_ly(x = ~date, y = ~investment.growth, type = "scatter", mode = "lines") 

returns %>% 
    tq_performance(Ra = close_ret,
                   performance_fun = SharpeRatio,
                   Rf = 0.04/12)

returns %>% 
    group_by(symbol) %>%
    tq_performance(Ra = close_ret,
                   performance_fun = SharpeRatio,
                   Rf = 0.04/12)

returns %>% 
    summarise(min = min(date), .by = symbol)

# test forecast

lgb_forecast <- modeltime_table(wflw_fit_lgb_tuned) %>% 
    modeltime_refit(data_prepared_dt_filter) %>% 
    modeltime_forecast(
        new_data    = forecast_dt,
        actual_data = data_prepared_dt_filter,
        keep_data   = T,
        conf_by_id  = T
    )

lgb_forecast %>% filter(symbol == "ADSK")

# test new trend features
test_data <- data_prepared_dt[symbol == "ADSK"]

test_data %>% 
    select(Return_fwd_21,date, contains("MACD")) %>% 
    pivot_longer(contains("MACD")) %>% 
    mutate(std = (value-mean(value))/sd(value),
           .by = name) %>% 
    mutate(ret_std = (Return_fwd_21-mean(Return_fwd_21, na.rm = T))/sd(Return_fwd_21, na.rm = T)) %>% 
    ggplot(aes(x = date, y = std, color = name))+
    geom_line(show.legend = F)+
    geom_line(aes(x = date,y = ret_std, color = "red"),show.legend = F)+
    facet_wrap(~name, scales = "free")

test_data %>% 
    select(Return_fwd_21,date, contains("MACD")) %>% 
    pivot_longer(contains("MACD")) %>% 
    mutate(std = (value-mean(value))/sd(value),
           .by = name) %>% 
    mutate(ret_std = (Return_fwd_21-mean(Return_fwd_21, na.rm = T))/sd(Return_fwd_21, na.rm = T)) %>% 
    dplyr::group_by(name) %>% 
    cor_test(ret_std,std)

library(ranger)
test_rf <- ranger(Return_fwd_21 ~ ., test_data %>% 
           select(Return_fwd_21,contains("MACD")) %>% 
           select(-contains("trading")) %>% 
           filter(!is.na(Return_fwd_21)), 
       importance = "permutation") 

test_rf %>% 
    importance() %>% 
    enframe() %>% 
    arrange(desc(value))

experiment <- copy(test_data)

experiment[, ":=" (
    Close_macd_long_trend           = Close_macd_long/EMA(Close_macd_long,21),
    Close_macd_long_signal_trend    = Close_macd_long_signal/EMA(Close_macd_long_signal,21),
    Close_macd_short_trend          = Close_macd_short/EMA(Close_macd_short,21),
    Close_macd_short_signal_trend   = Close_macd_short_signal/EMA(Close_macd_short_signal,21),
    Close_macd_long_trading_signal  = Close_macd_long-Close_macd_long_signal,
    Close_macd_short_trading_signal = Close_macd_short-Close_macd_short_signal
)]

experiment %>% select(Return_fwd_21, contains("MACD")) %>% cor_test(Return_fwd_21)


exp_rf <- ranger(Return_fwd_21 ~ ., experiment %>% 
                     select(Return_fwd_21,contains("MACD")) %>% 
                     #select(-contains("trading")) %>% 
                     filter(!is.na(Return_fwd_21)), 
                 importance = "permutation")

exp_rf %>% 
    importance() %>% 
    enframe() %>% 
    arrange(desc(value))

pred <- predict(test_rf, test_data)
pred

cbind(test_data$Return_fwd_21, test_data$date, pred$predictions) %>% tail(5)
bind_cols(ret = test_data$Return_fwd_21, date = test_data$date, pred = pred$predictions)

# test MACD time lengths ----
prices_features_dt[, Close_macd_long := MACD(close, 63, 252, 21)[, "macd"], keyby = symbol] # base 63, 252, 21
prices_features_dt[, Close_macd_long_signal := MACD(close, 63, 252,21)[, "signal"], keyby = symbol]
prices_features_dt[, Close_macd_short := MACD(close, 18, 36, 18, list(list(EMA, wilder=TRUE),list(EMA, wilder=TRUE),list(EMA, wilder=TRUE)))[, "macd"], keyby = symbol] # 18, 36, 18
prices_features_dt[, Close_macd_short_signal := MACD(close, 18, 36, 18, list(list(EMA, wilder=TRUE),list(EMA, wilder=TRUE),list(EMA, wilder=TRUE)))[, "signal"], keyby = symbol]
prices_features_dt[, ":=" (
    Close_macd_long_trend           = frollmean(Close_macd_long,63),
    Close_macd_long_signal_trend    = frollmean(Close_macd_long_signal,63),
    Close_macd_short_trend          = frollmean(Close_macd_short,63),
    Close_macd_short_signal_trend   = frollmean(Close_macd_short_signal,63),
    Close_macd_long_trading_signal  = Close_macd_long-Close_macd_long_signal,
    Close_macd_short_trading_signal = Close_macd_short-Close_macd_short_signal
), keyby = symbol]

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

train_pool <- catboost.load_pool(data = train %>% select(-Return_fwd_21,-date, -symbol), label = train$Return_fwd_21)
test_pool <- catboost.load_pool(data = test %>% select(-Return_fwd_21,-date, -symbol), label = test$Return_fwd_21)
valid_pool <- catboost.load_pool(data = valid %>% select(-Return_fwd_21,-date, -symbol), label = valid$Return_fwd_21)

# modeling
set.seed(121)
start <- Sys.time()
model  <- catboost.train(train_pool,  test_pool,
                         params = list(loss_function = 'RMSE',
                                       iterations    = 3000
                                       , early_stopping_rounds = 20
                                       , thread_count          = parallelly::availableCores(omit = 1)
                                       #, od_type       = 'Iter'   # Early stopping
                                       #, od_wait       = 20
                                       , verbose = 500
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

# test set accuracy
test_preds     <- predict(model, test_pool)
test_predicted <- bind_cols(test, pred = test_preds)
metrics(test_predicted, truth = Return_fwd_21, estimate = pred)

# validate out of sample prediction accuracy
preds     <- predict(model, valid_pool)
predicted <- bind_cols(valid, pred = preds)
metrics(predicted, truth = Return_fwd_21, estimate = pred)

# Accy vs return ----
ret_acc_merge <- merge(
    prices_features_dt[!is.na(Return_fwd_21),][date >= max(date)-years(2),
                                              .(avg_ret = mean(Return_fwd_21),
                                                sd_ret  = sd(Return_fwd_21),
                                                last_ret = sum(fifelse(date == max(date),1,0)*Return_fwd_21)), 
                                              keyby = symbol],
    acc_by_symbol) |> 
    mutate(acc_rank = rank(rmse),
           sharpe = (avg_ret-0.003)/sd_ret)

ret_acc_merge |> cor_test(rmse,rsq, avg_ret:last_ret) |> filter(var1 != var2) |> arrange(desc(abs(cor)))

ret_acc_merge |> 
    select(rmse, avg_ret, sd_ret, last_ret) |> 
    pivot_longer(-rmse) |> 
    ggplot(aes(y = rmse, x = value, color = name))+
    geom_point(alpha = 0.8)+
    geom_smooth(method = "lm")+
    facet_wrap(~name, scales = "free_x")+
    theme_minimal()

ret_acc_merge |> 
    plot_ly(x = ~acc_rank, y = ~sharpe)

ret_acc_merge |> 
    select(sharpe, avg_ret, sd_ret) |> 
    pivot_longer(-sharpe) |> 
    ggplot(aes(y = sharpe, x = value, color = name))+
    geom_point(alpha = 0.8)+
    geom_smooth(method = "lm")+
    facet_wrap(~name)+
    theme_minimal()

library(GGally)

ret_acc_merge |> 
    select(where(is.numeric), -acc_rank) |>
    ggpairs()

# calculate the number of items with the best accuracy and a minimum annual return of 20%
ret_acc_merge[acc_rank <= (max(acc_rank)*0.35) & avg_ret > 0.015,] |> get_summary_stats(avg_ret)
ret_acc_merge[acc_rank <= (max(acc_rank)*0.4) & avg_ret > 0.015,] |> get_summary_stats(avg_ret)

# test if it is better to use last or average prediction

# SAR testing ----
# https://www.linnsoft.com/techind/parabolic-sar-sar

# get some random symbols
set.seed(101)
symbols <- tq_index("SP500") |> 
    slice_sample(prop = 0.3) |> 
    pull(symbol)

# preprocess
test_data <- prices_features_dt[symbol %in% symbols &
                                    !is.na(Close_macd_long_signal_trend),
                                select(.SD,
                                       -(open:adjusted),
                                       close, # need to add close back so can calculate with the SAR ratio
                                       -Return_fwd_5, -Return_fwd_10,
                                       -contains("_lag_"),
                                       -contains("_lead_"))] %>% 
    group_by(symbol) %>%
    tk_augment_fourier(date, .periods = c(252), .K = 1) %>%
    ungroup() %>% 
    # tk_augment_timeseries_signature(date) %>% 
    # select(-matches("(.xts$)|(.iso$)|(.lbl$)|(.hour)|(.minute)|(.second)|(.am.pm)")) %>% 
    # select(-index.num, -diff, -year) %>% 
    setDT()

# cor test to returns
test_data[, .SD,.SDcols = sapply(test_data, is.numeric)] |> cor_test(Return_fwd_21) |> arrange(desc(cor)) |> View()

test_data |> plot_ly(x = ~date, y = ~SAR, color = ~symbol, mode = 'lines')
test_data |> plot_ly(y = ~Return_fwd_21, x = ~SAR, color = ~symbol)

test_data |> plot_ly(y = ~Return_fwd_21, x = ~(SAR/close), color = ~symbol)
test_data |> plot_ly(y = ~Return_fwd_21, x = ~(SAR - frollmean(SAR, 63))/frollsd(SAR, 63), color = ~symbol)

test_data[,.(SAR,
             SAR-close, 
             shift(SAR,1)-shift(close,1),
             Return_fwd_21)] |> cor_test(Return_fwd_21)

test_data[,.(SAR,
             ratio = SAR/close, 
             ratio_shift = shift(SAR,1)/shift(close,1),
             ratio_compare = (SAR/close) - (shift(SAR,1)/shift(close,1)), 
             sar_rollsd_3 = (SAR - frollmean(SAR, 63))/frollsd(SAR, 63),
             ratio_rollsd_3 = ((SAR/close) - frollmean(SAR/close, 63))/frollsd(SAR/close, 63),
             ratio_rollsd_6 = ((SAR/close) - frollmean(SAR/close, 126))/frollsd(SAR/close, 126),
             sar_rollsd_1 = (SAR - frollmean(SAR, 21))/frollsd(SAR, 21),
             sar_rollsd_6 = (SAR - frollmean(SAR, 126))/frollsd(SAR, 126),
             Return_fwd_21)] |> cor_test(Return_fwd_21) |> 
    arrange(desc(cor))

model_data <- test_data[,.(Close_roc_0_1_std_63,
                           date_sin252_K1,
                           SAR,
                           ratio = SAR/close,
                           # ratio_shift = shift(SAR,1)/shift(close,1),
                           # ratio_compare = (SAR/close) - (shift(SAR,1)/shift(close,1)),
                           # sar_rollsd_1 = (SAR - frollmean(SAR, 21))/frollsd(SAR, 21),
                           sar_rollsd_3 = (SAR - frollmean(SAR, 63))/frollsd(SAR, 63),
                           sar_rollsd_6 = (SAR - frollmean(SAR, 126))/frollsd(SAR, 126),
                           sar_rollsd_12 = (SAR - frollmean(SAR, 252))/frollsd(SAR, 252),
                           # ratio_rollsd_3 = ((SAR/close) - frollmean(SAR/close, 63))/frollsd(SAR/close, 63),
                           # ratio_rollsd_6 = ((SAR/close) - frollmean(SAR/close, 126))/frollsd(SAR/close, 126),
                           Return_fwd_21)] |> na.omit()
set.seed(101)
split <- initial_split(model_data)

library(ranger)
options(ranger.num.threads = 6)

ranger <- ranger(Return_fwd_21 ~ .,
                 model_data,
                 importance = 'permutation')

ranger

ranger |> importance() |> enframe() |> arrange(desc(value))

library(xgboost)
library(tidymodels)

set.seed(101)
xgb <- boost_tree(
    "regression",
    trees     = 3000,
    stop_iter = 20) |> 
    set_engine("xgboost", validation = 0.2, nthread = 4) |> 
    fit(Return_fwd_21 ~ ., training(split)) #
    # fit_resamples(Return_fwd_21 ~., vfold_cv(model_data, v = 5))

extract_fit_engine(xgb) |> xgb.importance() |> arrange(desc(Gain))

augment(xgb, testing(split)) |> metrics(truth = Return_fwd_21, estimate = .pred)

library(shapviz)

fit <- extract_fit_engine(xgb)
xnames <- variable.names(fit)

x_explain <- na.omit(testing(split) |> select(all_of(xnames)) |> slice_sample(prop = 0.5))

shap <- shapviz(fit, 
                X_pred = data.matrix(x_explain), 
                X = x_explain)

shap |> 
    sv_importance(show_numbers = TRUE, "beeswarm", alpha = 0.7) 

shap |> 
    sv_importance(show_numbers = TRUE, "bar", alpha = 0.7) 

shap |> 
    sv_dependence(xnames, share_y = TRUE)

library(glmnet)
x = model_data[,!c("Return_fwd_21")] |> na.omit()
y = model_data[!is.na(sar_rollsd_12), Return_fwd_21]

glm <- glmnet(x = x,
              y = y)
glm
glm

coef(glm, s = 0.001)

# compare stocks purchased in sept vs those with best accuracy or those with best final prediction ----
options(scipen = 99)

test_from <- "2026-09-03"

# import data from saved data ----
model_ensemble_final_forecast <-readRDS("01_save_data/01_saved_forecasts/2026-09-03_model_ensemble_final_forecast.rds")
acc_by_symbol <- readRDS("02_models/2026-09-03_acc_by_symbol.rds")

# * accuracy by symbol ----
forecast_acc_symbol <- model_ensemble_final_forecast %>% 
    #filter(date == max(date)) %>% 
    merge(acc_by_symbol)

# what was actually picked
pred_avg <- forecast_acc_symbol %>% 
    filter(.key == 'prediction') %>% 
    summarise(mean_pred = mean(.value), .by = symbol) %>% 
    arrange(desc(mean_pred)) %>% 
    merge(acc_by_symbol)  

stock_picks <- pred_avg %>% 
    slice_max(mean_pred, n = 10) %>% 
    arrange(symbol) |> 
    pull(symbol)

portfolio1 <- tq_get(stock_picks, from = test_from)

portfolio1 %>%
    ggplot(aes(x = date, y = close)) +
    geom_candlestick(aes(open = open, high = high, low = low, close = close),
                     colour_up = "darkgreen", colour_down = "darkred", 
                     fill_up  = "darkgreen", fill_down  = "darkred") +
    labs(y = "Closing Price", x = "") + 
    facet_wrap(~ symbol, scale = "free_y") +
    theme_tq()

portfolio1 %>%
    ggplot(aes(x = date, y = close)) +
    geom_line() +
    labs(y = "Closing Price", x = "") + 
    facet_wrap(~ symbol, scale = "free_y") +
    theme_tq()

returns1 <- portfolio1 %>%
    group_by(symbol) %>%
    tq_transmute(select = close,
                 mutate_fun = periodReturn,
                 period = 'daily',
                 col_rename = "close_ret") %>% 
    ungroup()


returns_xts1 <- returns1 %>%
    drop_na() %>% 
    pivot_wider(id_cols = date, names_from = symbol, values_from = close_ret) %>%
    as.xts()

returns1 %>% 
    tq_portfolio(assets_col   = symbol,
                 returns_col  = close_ret,
                 #weights      = wts,
                 col_rename   = "investment.growth",
                 wealth.index = T) %>%
    mutate(investment.growth = investment.growth * 10000) %>% 
    plot_ly(x = ~date, y = ~investment.growth, type = "scatter", mode = "lines+markers") |> 
    layout(title = "Actual portfolio")

returns1 %>% 
    group_by(symbol) %>%
    tq_performance(Ra = close_ret,
                   performance_fun = SharpeRatio,
                   Rf = 0.06/12)

base_return <- Return.portfolio(returns_xts1)
table.AnnualizedReturns(base_return)


# what would have been picked using the last forecast date
last_predictions <- forecast_acc_symbol %>%
    filter(.key == "prediction") |> 
    filter(date == max(date)) |> 
    slice_max(.value, n = 10) |> 
    select(symbol, .value, date)

stock_picks_last <- last_predictions |>
    pull(symbol)

setdiff(stock_picks,stock_picks_last)
setdiff(stock_picks_last,stock_picks)

portfolio_last <- tq_get(stock_picks_last, from = test_from)

portfolio_last %>%
    ggplot(aes(x = date, y = close)) +
    geom_candlestick(aes(open = open, high = high, low = low, close = close),
                     colour_up = "darkgreen", colour_down = "darkred", 
                     fill_up  = "darkgreen", fill_down  = "darkred") +
    labs(y = "Closing Price", x = "") + 
    facet_wrap(~ symbol, scale = "free_y") +
    theme_tq()

portfolio_last %>%
    ggplot(aes(x = date, y = close)) +
    geom_line() +
    labs(y = "Closing Price", x = "") + 
    facet_wrap(~ symbol, scale = "free_y") +
    theme_tq()

returns_last <- portfolio_last %>%
    group_by(symbol) %>%
    tq_transmute(select = close,
                 mutate_fun = periodReturn,
                 period = 'daily',
                 col_rename = "close_ret") %>% 
    ungroup()


returns_xts_last <- returns_last %>%
    drop_na() %>% 
    pivot_wider(id_cols = date, names_from = symbol, values_from = close_ret) %>%
    as.xts()

returns_last %>% 
    tq_portfolio(assets_col   = symbol,
                 returns_col  = close_ret,
                 #weights      = wts,
                 col_rename   = "investment.growth",
                 wealth.index = T) %>%
    mutate(investment.growth = investment.growth * 10000) %>% 
    plot_ly(x = ~date, y = ~investment.growth, type = "scatter", mode = "linesmarkers") |> 
    layout(title = "Last date portfolio")

returns_last %>% 
    group_by(symbol) %>%
    tq_performance(Ra = close_ret,
                   performance_fun = SharpeRatio,
                   Rf = 0.06/12)

last_return <- Return.portfolio(returns_xts_last)

# pick based on lowest (best) rmse
# what was actually picked
pred_avg <- model_ensemble_final_forecast %>% 
    filter(.key == 'prediction') %>% 
    summarise(mean_pred = mean(.value), .by = symbol) %>% 
    select(symbol, mean_pred) |> 
    merge(acc_by_symbol)  

stock_picks_acc <- pred_avg %>% 
    filter(mean_pred > 0) |> 
    slice_min(rmse, prop = 0.2) |> slice_max(mean_pred, n = 10) |> 
    #slice_min(rmse, n = 10) |> 
    arrange(symbol) |> 
    pull(symbol)

setdiff(stock_picks_acc,stock_picks)

portfolio_acc <- tq_get(stock_picks_acc, from = test_from)

portfolio_acc %>%
    ggplot(aes(x = date, y = close)) +
    geom_candlestick(aes(open = open, high = high, low = low, close = close),
                     colour_up = "darkgreen", colour_down = "darkred", 
                     fill_up  = "darkgreen", fill_down  = "darkred") +
    labs(y = "Closing Price", x = "") + 
    facet_wrap(~ symbol, scale = "free_y") +
    theme_tq()

portfolio_acc %>%
    ggplot(aes(x = date, y = close)) +
    geom_line() +
    labs(y = "Closing Price", x = "") + 
    facet_wrap(~ symbol, scale = "free_y") +
    theme_tq()

returns_acc <- portfolio_acc %>%
    group_by(symbol) %>%
    tq_transmute(select = close,
                 mutate_fun = periodReturn,
                 period = 'daily',
                 col_rename = "close_ret") %>% 
    ungroup()


returns_xts_acc <- returns_acc %>%
    drop_na() %>% 
    pivot_wider(id_cols = date, names_from = symbol, values_from = close_ret) %>%
    as.xts()

returns_acc %>% 
    tq_portfolio(assets_col   = symbol,
                 returns_col  = close_ret,
                 #weights      = wts,
                 col_rename   = "investment.growth",
                 wealth.index = T) %>%
    mutate(investment.growth = investment.growth * 10000) %>% 
    plot_ly(x = ~date, y = ~investment.growth, type = "scatter", mode = "lines+markers") |> 
    layout(title = "Accuracy portfolio")

returns_acc %>% 
    group_by(symbol) %>%
    tq_performance(Ra = close_ret,
                   performance_fun = SharpeRatio,
                   Rf = 0.06/12)

acc_return <- Return.portfolio(returns_xts_acc)

table.AnnualizedReturns(base_return)
table.AnnualizedReturns(last_return)
table.AnnualizedReturns(acc_return)

# sp500 for comparison
sp <- tq_get("^GSPC", from = test_from)

sp_returns <- sp %>%
    tq_transmute(select = adjusted,
                 mutate_fun = periodReturn,
                 period = 'daily',
                 col_rename = "return_month") %>%
    ungroup()

sp_returns_monthly_xts <- sp_returns %>%
    #filter(date >= filter_date) %>%
    rename("GSPC" = "return_month") %>% 
    as.xts()

# unweighted portfolio returns
market_base <- Return.portfolio(sp_returns_monthly_xts)
table.AnnualizedReturns(market_base)

ports <- cbind(base_return,
               market_base,
               last_return, 
               acc_return) #,port_port_returns_ma,
colnames(ports) <- c("base","SP500","last.date","accuracy") #"ma_test",
table.AnnualizedReturns(ports, Rf = 0.05/12)
charts.PerformanceSummary(ports, Rf = 0.05/12)

# Reward to volatility measure ----
prices <- tq_get(c("NVDA","MMM","CBOE","HWM","CAH","CSCO","AAPL","HCA","QUAL"))
setDT(prices)

summary(prices)

prices[, ":=" (return_5 = (close/lag(close, 5))-1
               , return_21 = (close/lag(close, 21))-1
               , reward_to_volatility = ((close/lag(close, 5))-1) / frollsd(close, 63)
               , Close_cmo_28       = CMO(close, n = 28)), 
       keyby = symbol][,mean_reward_vol := frollmean(reward_to_volatility, 21), keyby = symbol]

prices

prices |> cor_test(return_21)

prices |> 
    ggplot(aes(y = return_21, x = reward_to_volatility, color = symbol))+
    geom_point(alpha = 0.8, position = "jitter", show.legend = F)+
    facet_wrap(~symbol, scales = "free")+
    geom_smooth(method = "gam", color = "gray50")

prices |> 
    ggplot(aes(y = return_21, x = return_5, color = symbol))+
    geom_point(alpha = 0.8, position = "jitter", show.legend = F)+
    facet_wrap(~symbol, scales = "free")+
    geom_smooth(method = "gam", color = "gray50")

prices |> 
    ggplot(aes(y = return_21, x = mean_reward_vol, color = symbol))+
    geom_point(alpha = 0.8, position = "jitter", show.legend = F)+
    facet_wrap(~symbol, scales = "free")+
    geom_smooth(method = "gam", color = "gray50")

lm1 <- lm(return_21 ~ mean_reward_vol, data = prices)
lm2 <- lm(return_21 ~ reward_to_volatility, data = prices)
lm3 <- lm(return_21 ~ reward_to_volatility + mean_reward_vol, data = prices)
lm4 <- lm(return_21 ~ reward_to_volatility + mean_reward_vol + Close_cmo_28, data = prices)

stargazer::stargazer(lm1, lm2, lm3, lm4, type = "text")

# ml testing
split <- initial_split(prices[!is.na(return_21) & !is.na(mean_reward_vol)], 0.7, strata = symbol)
test  <- testing(split)
train <- training(split)

# library(ranger)

ranger <- ranger(return_21 ~ reward_to_volatility + mean_reward_vol+ Close_cmo_28, data = train,
                 num.threads = 6,seed = 101,
                 importance = "impurity_corrected")
ranger

importance(ranger)

pred <- predict(ranger, test)

metrics(truth = return_21, estimate = predicted,cbind(test, predicted = pred$predictions))

set.seed(101)
xgb <- boost_tree("regression",
                  trees     = 2000,
                  stop_iter = 30) |> 
    set_engine("xgboost",
               nthread     =  -1, 
               validation  = 0.1) |> 
    fit(return_21 ~ reward_to_volatility + mean_reward_vol + Close_cmo_28, data = train)

augment(xgb, test) |> metrics(truth = return_21, estimate = .pred)

xgboost::xgb.importance(model = extract_fit_engine(xgb)) |> arrange(desc(Gain))


library(glmnet)
x = train[,.(reward_to_volatility, mean_reward_vol, Close_cmo_28)]
y = train[,return_21]

glm <- glmnet(x = x,
              y = y, alpha = 1)
glm

coef(glm, s = 0.0001)

predict(glm, as.matrix(test[,.(reward_to_volatility, mean_reward_vol, Close_cmo_28)]))

        