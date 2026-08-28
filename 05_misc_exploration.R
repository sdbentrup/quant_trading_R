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

