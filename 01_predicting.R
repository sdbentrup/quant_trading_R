# forecast stock returns
# try to predict returns 21-days forward. consider this paper:
# https://jfin-swufe.springeropen.com/articles/10.1186/s40854-024-00644-0
# https://www.sciencedirect.com/science/article/pii/S2666827025000143

# packages ----
# Time Series ML
library(tidymodels)
library(modeltime)
library(modeltime.ensemble)
library(modeltime.resample)
library(boostime)
library(finetune)
library(xgboost)
library(lightgbm)
library(prophet)

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

# Modeling and stock analytics
library(bonsai)
library(baguette)
library(PortfolioAnalytics)
library(data.table)
library(plotly)
library(rstatix)
library(shapviz)
library(ggthemes)
library(rvest) # for read_html()
library(TTR)
library(xts)
library(shapviz)

# enter dates ----
from <- today() - years(6)
test_start <- today() - years(2)


# tidyquant get symbols from sp500

# * Calibrate function ----
calibrate_and_plot <- function(..., type = 'testing', print = T){
    
    if (type == 'testing'){
        new_data = testing(splits)
    } else {
        new_data = training(splits)
    }
    
    calibration_tbl <- modeltime_table(...) %>% 
        modeltime_calibrate(new_data)
    
    print(calibration_tbl %>% modeltime_accuracy())
    
    if (print == T){calibration_tbl %>% 
            modeltime_forecast(
                new_data = new_data,
                actual_data = data_prepared_tbl
            ) %>% 
            plot_modeltime_forecast(.conf_interval_show = F)
    } else {
        return()
    }
}

# 1.0 Data ----
# * symbols for testing ----
# SP500
sp500 <- tq_index("SP500") #tq_index() options include "DOW"       "DOWGLOBAL" "SP400"     "SP500"     "SP600" 

# could also use tq_exchange("NYSE"), tq_exchange("AMEX") tq_exchange("NASDAQ")

sp500_symbols <- sp500 %>% 
  filter(symbol != "-" & !str_detect(company,"CL C")) %>% 
  slice_max(weight, n = 100) %>% 
  arrange(symbol) %>% 
  pull(symbol) 

# sp500_tbl    <- tq_get(x = sp500,
#                        get = "stock.prices",
#                        from = from,
#                        complete_cases = T) %>%
#     mutate(symbol = str_replace(symbol, "-","."))
# 
# setDT(prices_base)

symbols <- c("AAP","AAPL","ALB","CE","DLTR","INTC","JD","MOH","MPWR","MMM","MU",
             "NCLH","NVDA","NWL","QRVO","SEDG","WBA","ZTS",
             "BSX","CTAS","FI","FIS","GE","ICE","IRM","JNPR","LDOS","META",
             "TT","VTR","WELL","VRTX")

symbols_all <- union(sp500_symbols, symbols)

# * get stock price data ----
prices_base <- tq_get(unique(symbols_all), from = from) %>% 
  mutate(symbol = str_replace(symbol, "-",".")) %>% 
  setDT()

prices_base

prices_base[,.(.N),keyby = symbol] %>% arrange(N)

# * reduce data with filters ----

symbols_filtered_list <- prices_base[,.(count = .N,
                max = max(date),
                min = min(date)),
             keyby = symbol][count >= 252*5 & max == max(max),symbol]


# * create filtered dataset ----
prices_dt <- prices_base[symbol %in% symbols_filtered_list,]

# * Engineered indicators ----
# function for features
add_best_features <- function(prices_dt, price) {
  price_col <- prices_dt[[price]]
  
  # Calculate Indicators
  prices_dt[, Close_macd_50_200_30 := MACD(price_col, 50, 200, 30, maType = "EMA")[, "macd"]]
  prices_dt[, Close_macd_signal_50_200_30 := MACD(price_col, 50, 200, 30, maType = "EMA")[, "signal"]]
  prices_dt[, Close_macd_14_30_9 := MACD(price_col, 14, 30, 9, list(list(EMA, wilder=TRUE),list(EMA, wilder=TRUE),list(EMA, wilder=TRUE)))[, "macd"]]
  prices_dt[, Close_macd_signal_14_30_9 := MACD(price_col, 14, 30, 9, list(list(EMA, wilder=TRUE),list(EMA, wilder=TRUE),list(EMA, wilder=TRUE)))[, "signal"]]
  prices_dt[, Close_ema_10_norm  := EMA(price_col, n = 10) / price_col]
  prices_dt[, Close_ema_21_norm  := EMA(price_col, n = 21) / price_col]
  prices_dt[, Close_ppo_line_12_26 := ((EMA(price_col, 12) - EMA(price_col, 26)) / EMA(price_col, 26)) * 100]
  prices_dt[, Close_oscillator_7_14_28 := ultimateOscillator(prices_dt[, .(high, low, price_col)])]
  prices_dt[, Close_roc_0_1      := ROC(price_col, n = 1)]
  prices_dt[, Close_roc_0_5      := ROC(price_col, n = 5)]
  prices_dt[, Close_roc_0_21     := ROC(price_col, n = 21)]
  prices_dt[, Close_roc_0_1_rolling_std_win_63 := frollapply(Close_roc_0_1, 63, sd, fill = NA, align = "right")]
  prices_dt[, Close_natr_63      := ATR(prices_dt[,.(high, low, price_col)], n = 63)[,"atr"]/eval(quote(get(price)))]
  #prices_dt[, Close_rsi_10 := RSI(eval(quote(get(price))), n = 10)]
  prices_dt[, Close_rsi_21       := RSI(price_col, n = 21)]
  prices_dt[, Close_cmo_28       := CMO(price_col, n = 28)]
  prices_dt[, Close_rolling_mean_win_126 := standardize_vec(frollmean(price_col, 126))]
  prices_dt[, Close_rolling_std_win_126  := frollapply(price_col, 126, sd)]
  prices_dt[, Close_SNR_21       := SNR(prices_dt[, .(high, low, price_col)], n = 21)]
  prices_dt[, Close_rel_vol_14   := 100 - 100 / (1 + runSD(price_col, 14))]
  prices_dt[, Close_252_max_diff := price_col/frollapply(price_col, 252, max)]
  prices_dt[, Close_252_min_diff := price_col/frollapply(price_col, 252, min)]
  prices_dt[, Close_kst          := KST(price_col)[,"kst"]]
  prices_dt[, Close_kst_signal   := KST(price_col)[,"signal"]]
  prices_dt[, SAR                := SAR(prices_dt[,.(high, low)])]
  #prices_dt[, Close_kama         := KAMA(price_col)]

  prices_dt[, ":=" (intraday = high/low-1, day_return = open/close-1)]
  
  prices_dt[, Vol_WAP_norm := standardize_vec(VWAP(price_col, volume))]
  prices_dt[, Vol_OBV_norm := standardize_vec(OBV(price_col, volume))]
  
  prices_dt[, Vol_MFI_21 := MFI(prices_dt[,.(high, low, close)], volume, n = 21)]
  
  # Fast QS Momentum
  prices_dt[, `:=`(
    Close_lag_1   = data.table::shift(price_col, n = 1),
    Close_lag_21  = data.table::shift(price_col, n = 21),
    Close_lag_252 = data.table::shift(price_col, n = 252)
  )]
  
  prices_dt[, Close_fastqsmom_21_252_126 := ((Close_lag_21 / Close_lag_252 - 1) - (Close_lag_1 / Close_lag_21 - 1))]
  
  # Z-score features
  prices_dt[, Close_zscore_126 := (price_col - Close_rolling_mean_win_126) / Close_rolling_std_win_126]
  
  # Efficiently compute lead values & forward returns without loops
  T <- c(5, 10, 21)
  lead_cols   <- paste0("Close_lead_", T)
  return_cols <- paste0("Return_fwd_", T)
  
  prices_dt[, (lead_cols) := lapply(T, function(t) data.table::shift(price_col, -t))]
  prices_dt[, (return_cols) := lapply(lead_cols, function(col) (get(col) / price_col) - 1)]
  
  return(prices_dt)
}

# Apply function
prices_features_list <- future_lapply(split(prices_dt, by = "symbol"), 
                                      add_best_features,
                                      price = "close")

# Combine the results back into a `data.table`
prices_features_tbl <- rbindlist(prices_features_list, use.names = TRUE, fill = TRUE)

# Charts ----
# consider correlations between features
prices_features_tbl %>% 
  select(where(is.numeric), -(open:adjusted), -contains("_lag_"),
         -contains("_lead_")) %>% 
  cor_test(Return_fwd_21) %>% 
  filter(var1 != var2) %>% 
  plot_ly(x = ~cor,
          y = ~fct_reorder(var2, cor, .desc = F),
          type = "bar",
          alpha = 0.7) %>% layout(yaxis = list(title = ''))

# library(GGally)
# 
# ggpairs(prices_features_tbl %>% 
#           select(where(is.numeric), 
#                  -(open:adjusted), 
#                  -contains("_lag_"),
#                  -contains("_lead_")),
#   lower = list(
#     continuous = "smooth"
#   )) + theme_tq()

# chart price features to see if they need to be scaled
prices_features_tbl %>% 
    filter(symbol %in% symbols[c(7:27)]) %>% droplevels() %>% 
  select(symbol, starts_with("Close_"),  -contains("_lag_"),
         -contains("_lead_")) %>% 
  pivot_longer(-symbol) %>% 
  filter(!is.na(value)) %>% 
  ggplot(aes(y = value, x = symbol, fill = symbol, color = symbol))+
  geom_boxplot(alpha = 0.7, show.legend = F)+
  facet_wrap(~name, scales = "free_y")+
  theme_clean()+
  theme(axis.text.x = element_text(angle = 30, hjust = 1),
        strip.background = element_rect("#0076BA"),
        strip.text = element_text(color = "white"))

prices_features_tbl %>% 
    filter(symbol %in% symbols[c(7:27)]) %>% droplevels() %>% 
  select(symbol, starts_with("Vol_")) %>% 
  pivot_longer(-symbol) %>% 
  filter(!is.na(value)) %>% 
  ggplot(aes(y = value, x = symbol, fill = symbol, color = symbol))+
  geom_boxplot(alpha = 0.7, show.legend = F)+
  facet_wrap(~name, scales = "free_y")+
  theme_clean()+
  theme(axis.text.x = element_text(angle = 30, hjust = 1),
        strip.background = element_rect("#0076BA"),
        strip.text = element_text(color = "white"))

prices_features_tbl[symbol %in% symbols[c(1:30)],] %>% 
    plot_ly(x = ~Close_kst, y = ~Return_fwd_21, type = 'scatter', mode = 'markers')

prices_features_tbl[symbol %in% symbols[c(1:30)],][,.(sar = standardize_vec(SAR),Return_fwd_21), keyby = .(symbol)] %>% 
    plot_ly(x = ~sar, y = ~Return_fwd_21, type = 'scatter', mode = 'markers')

prices_features_tbl %>% 
    filter(symbol %in% symbols[c(1:50)]) %>% droplevels() %>% 
    select(symbol, SAR, Close_kst, Close_kst_signal) %>% 
    pivot_longer(-symbol) %>% 
    filter(!is.na(value)) %>% 
    ggplot(aes(y = value, x = symbol, fill = symbol, color = symbol))+
    geom_boxplot(alpha = 0.7, show.legend = F)+
    facet_wrap(~name, scales = "free_y")+
    theme_clean()+
    theme(axis.text.x = element_text(angle = 30, hjust = 1),
          strip.background = element_rect("#0076BA"),
          strip.text = element_text(color = "white"))

prices_features_tbl %>% 
    select(symbol, date, adjusted) %>% 
    pivot_longer(adjusted) %>% 
    plot_time_series(date, value, .color_var = name, 
                     .smooth = F, .facet_vars = symbol,
                     .facet_ncol = 3)

prices_features_tbl %>% 
  plot_ly(y = ~Return_fwd_21, x = ~Close_macd_14_30_9, type = "scatter")

# earnings data ----

# * function for earnings for multiple symbols ----
get_earnings_data <- function(symbol){
    
    #symbol <- str_replace(symbol,"-",".")
    
    # create link with the symbol and read the page
    link <- str_glue("https://www.alphaquery.com/stock/{symbol}/earnings-history")
    
    earnings_page <- read_html(link)
    
    #earnings_page
    
    # convert the table to a data.table. works on the alphaquery not the zacks
    earnings_table <- html_element(earnings_page, "table") %>% 
        html_table(header = T, trim = T) %>% 
        setDT()
    
    # format text to dates and numbers
    earnings_table[,':=' (symbol = symbol,
                          date = as.Date(`Announcement Date`),
                          est_eps = str_remove(`Estimated EPS`,pattern = "\\$") %>% as.numeric(),
                          act_eps = str_remove(`Actual EPS`,pattern = "\\$") %>% as.numeric())]
    
    # shift saturday by 2 days, sunday by 1 day
    earnings_table[,date := fcase(wday(date) == 7, date + days(2),
                                  wday(date) == 1, date + days(1),
                                  default = date)]
    
    # for cases where an estimate eps wasn't given, use the actual eps
    earnings_table[,est_eps := fifelse(is.na(est_eps) & !is.na(act_eps), act_eps, est_eps)]
    
    #earnings_table <- earnings_table[date >= from-months(3),]
    
    # arrange by date ascending to allow for date calculations
    setorder(earnings_table, date)
    
    # engineer features in eps
    # ratio shows if eps is higher or lower than estimated
    # ann_number splits the announcements this is used later for calculating the days between announcements
    earnings_table[,":=" (eps_surprise = act_eps-est_eps,
                          eps_growth = act_eps-data.table::shift(act_eps, 1, type = "lag"),
                          #announcement = "Y",
                          ann_nbr = 1:.N)]
    
    earnings_table <- earnings_table[,select(.SD, -(`Announcement Date`:`Actual EPS`))]
}

# Use lapply to apply the function to each symbol
earnings_data_list <- future_lapply(unique(prices_features_tbl$symbol), 
                                    get_earnings_data)

# Combine the results into a single data.table (if applicable)
earnings_data <- rbindlist(earnings_data_list, fill = TRUE)

setkey(earnings_data, symbol, date)

# * merge eps to prices ----
# prices_fill_tbl %>% count(symbol)

# this creates NAs where there were no earnings
# we will fill these in later

prices_earnings_tbl <- merge(prices_features_tbl,
                             earnings_data,
                             by = c("date","symbol"),
                             all.x = T)

setorderv(prices_earnings_tbl, cols = c("symbol","date"))

prices_earnings_tbl %>% 
    filter(!is.na(est_eps))

prices_earnings_tbl %>% 
  filter(!is.na(est_eps) & symbol == "NVDA" )

prices_earnings_tbl %>% 
    filter(is.na(est_eps))

prices_earnings_tbl %>% 
    filter(symbol == "WBA" & date == "2015-09-08")

# * fill earnings ----
# discount the earnings later by the time with an interaction term
fill_cols <- patterns("eps|ann_nbr", cols = colnames(prices_earnings_tbl))

prices_earnings_tbl[, (fill_cols) := lapply(.SD, function(x) nafill(x, "locf")), 
                    .SDcols = fill_cols, 
                    by = symbol]

setnafill(prices_earnings_tbl, "const", 0, cols = "ann_nbr")

# this is used later as an interaction variable to see if the time since the announcement affects prediction
prices_earnings_tbl[,":=" (eps_ann_days = round(as.numeric(difftime(date,
                                                               min(date),
                                                               units = "days")),0),
                           pe_ratio = act_eps/close),
                    keyby = .(symbol, ann_nbr)]

summary(prices_earnings_tbl)

# Hidden Markov model ----

# * function ----
library(fHMM)
hidden_markov <- function(dt){
  
  data <- dt
  
  # 2 state model
  controls <- set_controls(
    states      = 3,
    sdds        = "t",
    file        = data,
    date_column = "date",
    data_column = "close",
    # file        = as.data.frame(prices_base),,
    logreturns  = TRUE,
    # from        = "2010-01-01",
    # to          = "2024-12-01",
    runs        = 60
  )
  
  data_hmm <- fHMM::prepare_data(controls) # parsnip also has a prepare_data function
  
  #summary(data_2)
  
  model_hmm <- fit_model(data_hmm, seed = 101, ncluster = 4)
  model_hmm <- decode_states(model_hmm)
  
  state <- model_hmm$decoding
}

# * apply to data ----
tic()
prices_earnings_tbl[, state := hidden_markov(.SD), by = symbol, .SDcols = c("date", "close")]
toc()

# Macroeconomic data ----
# * load macroeconomic data ----

# start the from data 3 months earlier to try to get an announcement before
# the start of the prices data
indicators <- c("GDPC1", # real gdp
             "CPILFESL", #core cpi
             "UNRATE", # unemployment rate
             "FEDFUNDS" # fed funds effective rate
)

economic_data <- tq_get(indicators, get = "economic.data", from = from-days(90))
setDT(economic_data)

# * data.table transformations ----
# change name of price because I don't like it
setnames(economic_data, "price","value")

# create the change indicator. some are already % so use an absolute change
economic_data[,":=" (change = value-data.table::shift(value, 1, "lag")), keyby = symbol]

# cast long to wide
economic_data_cast <- economic_data[,dcast(.SD,date ~ symbol, value.var = c("value","change"))]

# add annoucement number. this is useful for splitting and making date calculations from the
# last announcement
economic_data_cast[,econ_ann := 1:.N]

# pad so that we can fill before joining later
# pad to today so when it joins there is a full data set
economic_data_pad <- economic_data_cast %>% 
  pad_by_time(date, 
              .by = "day", 
              .fill_na_direction = "down", 
              .end_date = today()) %>% 
  setDT()

# calculate dates from last announcement
economic_data_pad[,econ_ann_days := round(as.numeric(difftime(date,
                                                              min(date),
                                                              units = "days")),0),
                  keyby = .(econ_ann)]  

# economic_data_pad

economic_data_final <- economic_data_pad[,.(date, 
                                            value_FEDFUNDS, 
                                            value_UNRATE,
                                            change_FEDFUNDS,
                                            change_UNRATE,
                                            change_CPILFESL, 
                                            change_GDPC1,
                                            econ_ann_days)]

economic_data_final[,unrate_over := fifelse(value_UNRATE > frollmean(value_UNRATE, n = 300),1,0)]

# * join economic data to prices ----
prices_earnings_econ_tbl <- merge(prices_earnings_tbl, 
                                  economic_data_final, 
                                  all.x = T,
                                  by = "date")

# Charts ----
prices_earnings_econ_tbl %>% 
  select(Return_fwd_21, eps_surprise, eps_growth, value_UNRATE, value_FEDFUNDS, unrate_over) %>% 
  cor_test() %>% 
  filter(var1 != var2) %>% 
  plot_ly(z = ~cor,
          x = ~var1,
          y = ~var2,
          #color = ~var1,
          type = "heatmap",
          alpha = 0.7) %>% layout(yaxis = list(title = ''))

prices_earnings_econ_tbl %>% 
  filter(symbol == "NVDA") %>% 
  plot_ly(x = ~date,
          y = ~Return_fwd_21,
          color = ~unrate_over,
          type = "scatter")


prices_features_tbl %>% 
  select(symbol, contains("eps")) %>% 
  pivot_longer(-symbol) %>% 
  filter(!is.na(value)) %>% 
  ggplot(aes(y = value, x = symbol, fill = symbol, color = symbol))+
  geom_boxplot(alpha = 0.7, show.legend = F)+
  facet_wrap(~name, scales = "free_y")+
  theme_clean()+
  theme(axis.text.x = element_text(angle = 30, hjust = 1),
        strip.background = element_rect("#0076BA"),
        strip.text = element_text(color = "white"))

prices_base %>% 
    select(date, adjusted, close) %>% 
    pivot_longer(-date) %>% 
    plot_time_series(date, value, .color_var = name, .smooth = F)

prices_base %>%
    ggplot(aes(x = date, y = adjusted)) +
    #geom_barchart(aes(open = open, high = high, low = low, close = close)) +
    geom_line()+
    geom_ma(color = "darkgreen", n = 200, linewidth = 1) +
    geom_ma(color = "red", n = 50, linewidth = 1) +
    geom_ma(color = "mediumblue", ma_fun = EMA, n = 50, linewidth = 1)+
    geom_ma(aes(volume = volume), color = "orange",ma_fun = EVWMA, n = 50, linewidth = 1)+
    coord_x_date(xlim = c("2020-01-01", "2024-01-31"))+
    #ylim = c(20, 30))
    theme_tq()

prices_base %>%
    tq_mutate(select = adjusted,mutate_fun = EMA, n = 30, col_rename = "ema_s") %>%
    tq_mutate(select = adjusted,mutate_fun = SMA, n = 200, col_rename = "sma_200") %>%
    plot_ly(x = ~date, y = ~adjusted, type = "scatter", mode = "lines", name = 'adjusted') %>%
    add_trace(y = ~ema_s, line = list(color = "green"), name = "ema 30") %>%
    add_trace(y = ~sma_200, line = list(color = "darkred"), name = "ema 200") %>%
    layout(xaxis = list(rangeslider = list(visible = T)))

prices_atr <- prices_base %>%
    tq_mutate(mutate_fun = ATR) %>%
    tq_mutate(select = adjusted, mutate_fun = WMA, n = 20, col_rename = "MA") %>%
    tq_mutate(mutate_fun = chaikinVolatility, col_rename = "chaikinVol")

ggplotly(prices_atr %>%
             select(date, close, atr, MA, chaikinVol) %>%
             pivot_longer(-date) %>%
             mutate(name = as.factor(name) %>% fct_relevel("close","MA")) %>%
             ggplot(aes(x = date, y = value, color = name))+
             geom_line(alpha = 0.8, linewidth = 1) +
             #geom_smooth(se = F)+
             #scale_color_viridis_d(option = "D")+
             scale_color_brewer(palette = "Dark2")+
             facet_wrap(~name, scales = "free_y", ncol = 1)+
             theme_bw()+
             theme(legend.position = "none"))

prices_atr %>% ggplot(aes(x = date,y = adjusted))+
    geom_line( color = "lightgreen")+
    geom_line(aes(y = trueHi), color = "orange")

# autocorrelation
prices_earnings_tbl %>% 
  filter(symbol == "NVDA") %>% 
    plot_acf_diagnostics(date, adjusted)

prices_base %>% 
    plot_acf_diagnostics(date, adjusted, .ccf_vars = volume, .show_ccf_vars_only = T)

prices_base %>% 
  filter(symbol == "NVDA") %>% 
    plot_seasonal_diagnostics(date, adjusted, .interactive = F)

# Create full data ----
full_data <- prices_earnings_econ_tbl %>% 
    
    # fix data issues 
    # could move the prices_earnings to this space...
    # select(date, symbol, Return_fwd_21) %>% 
    # group_by(symbol) %>% 
    # pad_by_time(date, .by = 'day', .pad_value = 0) %>% 
    # ungroup() %>% 
    
    # remove unneeded features
    select(-(open:adjusted),
           #-Return_fwd_5, -Return_fwd_10,
           -ann_nbr,
           -contains("_lag_"),
           -contains("_lead_")) %>%
    relocate(date, 
             symbol, 
             starts_with("Close_"), 
             starts_with("Vol_"), 
             starts_with("Return")) %>% 
    
    # global features
    # mutate(Return_fwd_21 = log1p(Return_fwd_21)) %>% 
    
    # groupwise features
    group_by(symbol) %>% 
  mutate(est_eps_norm = standardize_vec(est_eps),
         act_eps_norm = standardize_vec(act_eps)) %>% 
  #relocate(est_eps_norm:act_eps_norm, .after = adjusted) %>% 
    # future_frame(date, .length_out = 28, .bind_data = T) %>% 
    # group_split() %>% 
    # map(.f, add_best_features(df, "adjusted")) %>% 
    # bind_rows() %>% 
  ungroup() %>% 
  select(-est_eps, -act_eps) %>% 
    
    # lags / rolling / fourier features # fourier since it is panel data, different features by series
  mutate(symbol = as_factor(symbol)) %>% 
  group_by(symbol) %>% 
  arrange(date) %>% 
  tk_augment_fourier(date, .periods = c(5,21,63,252), .K = 2) %>%
  # tk_augment_timeseries_signature(date) %>% 
  # select(-matches("(.xts$)|(.iso$)|(.hour)|(minute)|(second)|(am.pm)"), -(hour:hour12), -diff) %>% 
    # tk_augment_holiday_signature(date,.holiday_pattern = "US_",.locale_set = "US", .exchange_set = "NYSE") %>% 
    # tk_augment_lags(.value = Close_roc_0_1, .lags = c(5, 10)) %>% 
    # tk_augment_slidify(
    #     Return_fwd_21_lag28,
    #     .f       = ~ mean(.x, na.rm = T),
    #     .period  = c(7, 28, 28*2),
    #     .partial = TRUE,
    #     .align   = 'center'
    # ) %>% 
  rowid_to_column(var = 'rowid') %>% 
  ungroup() %>% 
  setDT()

setorderv(full_data,cols = c("date","symbol"))

# add interaction variables
# full_data[,":=" (eps_growth_eps_ann_days       = eps_growth * eps_ann_days,
#                      eps_surprise_eps_ann_days     = eps_surprise * eps_ann_days,
#                      value_FEDFUNDS_econ_ann_days  = value_FEDFUNDS * econ_ann_days,
#                      value_UNRATE_econ_ann_days    = value_UNRATE * econ_ann_days,
#                      change_CPILFESL_econ_ann_days = change_CPILFESL * econ_ann_days)]

# * full data correlation visualization ----
full_data %>% 
  select(starts_with("Close_"), Return_fwd_21) %>% 
  drop_na() %>% 
  cor_test() %>% #View()
  plot_ly(x = ~var1,
          y = ~var2,
          z = ~cor,
          type = "heatmap",
          colors = c("darkgreen","white","darkgreen"))

full_data %>% 
  select(starts_with("date_"), Return_fwd_21) %>% 
  drop_na() %>% 
  cor_test() %>% #View()
  plot_ly(x = ~var1,
          y = ~var2,
          z = ~abs(cor),
          type = "heatmap",
          colors = c("white","darkgreen"))

# * save data for future use ----
write_rds(full_data, str_glue("01_save_data/{today()}_full_data.rds"))

# * remove unneeded data from environment ----
rm(prices_features_list)
rm(earnings_data)
rm(earnings_data_list)
rm(prices_earnings_tbl)
rm(prices_earnings_econ_tbl)
rm(prices_features_tbl)
rm(prices_base)
# not rm(prices_dt) since we can use this for the optimization
gc()

# 2.0 TIME SPLIT ----

# data prepared_tbl removes the lines missing data due to the rolling periods
data_prepared_tbl <- full_data %>% 
  filter(!is.na(Return_fwd_21))
    
forecast_tbl <- full_data %>% 
  filter(is.na(Return_fwd_21))    
    
splits <- data_prepared_tbl %>% 
  time_series_split(
    date_var = date,
    initial = "4 years",
    assess  = "60 days",
    cumulative = F
    )

# * plot time plan ----
splits %>% 
    tk_time_series_cv_plan() %>% 
    plot_time_series_cv_plan(.date_var = date, .value = Return_fwd_21)

# 3.0 RECIPE ----

# * Clean Training Set ----
# - With Panel Data, need to do this outside of a recipe
# - Transformation happens by group

train_cleaned <- training(splits) %>%
    group_by(symbol) %>% 
    #mutate(Return_fwd_21 = ts_clean_vec(Return_fwd_21, period = 7)) %>% 
    drop_na(Close_fastqsmom_21_252_126) %>% 
    ungroup()

# * plot splits ----
train_cleaned %>% 
    group_by(symbol) %>% 
    plot_time_series(
        date, Return_fwd_21,
        .facet_ncol = 3,
        .smooth = F,
        .trelliscope = T
    )

# * Recipe Specification ----

recipe_spec <- recipe(Return_fwd_21 ~ ., data = train_cleaned) %>%
  update_role(rowid, new_role = 'indicator') %>% 
  update_role(symbol, new_role = 'symbol') %>%
  step_mutate_at(symbol, fn = droplevels) %>%
  step_timeseries_signature(date) %>% 
  step_rm(matches("(.xts$)|(.iso$)|(.hour)|(.minute)|(.second)|(.am.pm)")) %>% 
  step_rm(Return_fwd_5, Return_fwd_10) %>%
  step_normalize(date_index.num,date_year) %>% 
  # step_normalize(index.num,year) %>% 
  step_dummy(all_nominal_predictors(), one_hot = T, keep_original_cols = F) %>%
  step_zv(all_predictors()) %>% 
  step_nzv(all_predictors(), unique_cut = 0.02) %>% 
  step_interact(~eps_growth:eps_ann_days) %>%
  step_interact(~eps_surprise:eps_ann_days) %>%
  step_interact(~value_FEDFUNDS:econ_ann_days) %>%
  step_interact(~value_UNRATE:econ_ann_days) %>%
  step_interact(~change_CPILFESL:econ_ann_days) %>%
    step_interact(~unrate_over:econ_ann_days)
  # step_corr(all_numeric_predictors(), threshold = 0.99)
  #step_pca(all_numeric_predictors())

recipe_spec %>% prep() %>% bake(new_data = train_cleaned %>% slice_sample(prop = 0.3)) %>% glimpse()

# 4.0 HYPERPARAMETER TUNING MODELS ---- 

# * RESAMPLES - K-FOLD ----- 

set.seed(69)
resamples_kfold <- train_cleaned %>% vfold_cv(v = 6)

# * plot resampling folds ----
resamples_kfold %>% 
    tk_time_series_cv_plan() %>% 
    plot_time_series_cv_plan(date, Return_fwd_21, .facet_ncol = 2)

# * Parallel Processing ----

# plan(strategy = cluster,  workers = 4)

# ncores <- parallelly::availableCores()
# plan(
#   strategy = cluster,
#   workers  = parallel::makeCluster(n_cores)
# )

# cl <- parallel::makeCluster(2, timeout = 60)
# plan(cluster, workers = cl)


# registerDoFuture()
# n_cores <- parallel::detectCores()
# plan(
#   strategy = cluster,
#   workers  = parallel::makeCluster(n_cores)
# )


options(future.globals.maxSize = 1.5 * 1024^3)  # Set to 1.5 GiB

plan(multisession, workers = 4)

plan(sequential)

# * LightGBM TUNE ----

# ** Tunable Specification

model_spec_lgb_tune <- boost_tree(
  "regression",
  mtry           = tune(),
  trees          = tune(),
  min_n          = tune(),
  tree_depth     = tune(),
  learn_rate     = tune(),
  #loss_reduction = 0.01,#tune()
  stop_iter      = 30) %>% 
  set_engine('lightgbm', counts = F, validation = 0.2)

wflw_spec_lgb_tune <- workflow() %>% 
  add_model(model_spec_lgb_tune) %>% 
  add_recipe(recipe_spec %>% step_rm(date))
  #add_recipe(recipe_spec %>% update_role(date, new_role = 'indicator'))

# ** Tuning
set.seed(69)
tic()
tune_results_lgb <- wflw_spec_lgb_tune %>% 
  tune_race_anova(
    resamples = resamples_kfold,
    param_info = extract_parameter_set_dials(wflw_spec_lgb_tune) %>% 
      update(learn_rate = learn_rate(range = c(0.1, 0.5), trans = NULL),
             trees      = trees(range = c(50,3000)),
             mtry       = mtry_prop(range = c(0.1,0.7))
      ),
    grid = 6,
    control = control_race(verbose = T, allow_par = T)
  )
toc()

# ** Results

tune_results_lgb %>% 
  show_best(metric = "rmse", n = Inf)

tune_results_lgb %>% 
  show_best(metric = "rsq", n = Inf)

# ** Finalize

wflw_fit_lgb_tuned <- wflw_spec_lgb_tune %>% 
  finalize_workflow(select_best(tune_results_lgb, metric = "rmse")) %>% 
  fit(train_cleaned)

calibrate_and_plot(wflw_fit_lgb_tuned, print = F)

modeltime_table(wflw_fit_lgb_tuned) %>% 
    modeltime_calibrate(new_data = testing(splits) %>% 
                            filter(symbol == "LLY")) %>% 
    modeltime_forecast(actual_data = data_prepared_tbl %>% 
                           filter(symbol == "LLY"), 
                       new_data = testing(splits) %>% 
                           filter(symbol == "LLY")
    ) %>% 
    plot_modeltime_forecast()

gc()

# * XGBOOST TUNE ----

# ** Tunable Specification
model_spec_xgboost_tune <- boost_tree(
    "regression",
    mtry           = tune(),
    trees          = tune(),
    min_n          = tune(),
    tree_depth     = tune(),
    learn_rate     = tune(),
    loss_reduction = 0.0005,#tune(),
    stop_iter      = 30
    ) %>% 
    set_engine('xgboost', 
               counts = F, 
               nthread =  -1, 
               tree_method = "hist",
               validation = 0.2)

wflw_spec_xgboost_tune <- workflow() %>% 
  add_model(model_spec_xgboost_tune) %>% 
  add_recipe(recipe_spec %>% step_rm(date))
    # add_recipe(recipe_spec %>% update_role(date, new_role = 'indicator'))

# ** Tuning

set.seed(69)
tic()
tune_results_xgboost <- wflw_spec_xgboost_tune %>% 
    tune_race_anova(
        resamples = resamples_kfold,
        param_info = extract_parameter_set_dials(wflw_spec_xgboost_tune) %>% 
          update(learn_rate = learn_rate(range = c(0.1, 0.5), trans = NULL),
                 trees      = trees(range = c(100,2500)),
                 mtry       = mtry_prop(range = c(0.1,0.7))
                 ),
        grid = 6,
        control = control_race(verbose = T, allow_par = T)
    )
toc()

# ** Results

tune_results_xgboost %>% 
    show_best(metric = "rmse", n = Inf)


# ** Finalize

tic()
wflw_fit_xgboost_tuned <- wflw_spec_xgboost_tune %>% 
    finalize_workflow(select_best(tune_results_xgboost, metric = "rmse")) %>% 
    fit(train_cleaned)
toc()

gc()

modeltime_table(wflw_fit_xgboost_tuned) %>% 
  modeltime_calibrate(new_data = testing(splits) %>% 
                        filter(symbol == "LLY")) %>% 
  modeltime_forecast(actual_data = data_prepared_tbl %>% 
                       filter(symbol == "LLY"), 
                     new_data = testing(splits) %>% 
                       filter(symbol == "LLY")
  ) %>% 
  plot_modeltime_forecast()

calibrate_and_plot(wflw_fit_xgboost_tuned, print =F)

# * EARTH TUNE ----

# ** Tunable Specification

model_spec_earth_tune <- mars(
    "regression",
    num_terms   = tune(),
    prod_degree = tune()
    ) %>% 
  set_engine("earth")

wflw_spec_earth_tune <- workflow() %>% 
  add_model(model_spec_earth_tune) %>% 
  add_recipe(recipe_spec %>% step_rm(date)) %>% 
  update_recipe(recipe_spec %>% step_rm(contains("sin")))
    # add_recipe(recipe_spec %>% update_role(date, new_role = 'indicator'))

# ** Tuning

tic()
set.seed(69)
tune_results_earth <- wflw_spec_earth_tune %>% 
    tune_grid(
        resamples = resamples_kfold,
        grid = 4,
        control = control_grid(verbose = T, allow_par = T)
    )
toc()

# ** Results
tune_results_earth %>% 
    show_best(metric = "rmse", n = Inf)

# ** Finalize

wflw_fit_earth_tuned <- wflw_spec_earth_tune %>% 
    finalize_workflow(select_best(tune_results_earth, metric = "rmse")) %>% 
    fit(train_cleaned)

gc()

# * glmnet TUNE ----

# ** Tunable Specification

model_spec_glmnet_tune <- linear_reg(
  mode = "regression",
  penalty = tune(), # higher penalty removes features faster
  mixture = tune(), 
) %>% 
  set_engine('glmnet')

wflw_spec_glmnet_tune <- workflow() %>% 
  add_model(model_spec_glmnet_tune) %>% 
  add_recipe(recipe_spec %>% step_rm(date))
# add_recipe(recipe_spec %>% update_role(date, new_role = 'indicator'))

# ** Tuning

tic()
set.seed(69)
tune_results_glmnet <- wflw_spec_glmnet_tune %>% 
  tune_grid(
    resamples = resamples_kfold,
    grid = 4,
    control = control_grid(verbose = T, allow_par = T)
  )
toc()

# ** Results
tune_results_glmnet %>% 
  show_best(metric = "rmse", n = Inf)


# ** Finalize

wflw_fit_glmnet_tuned <- wflw_spec_glmnet_tune %>% 
  finalize_workflow(select_best(tune_results_glmnet, metric = "rmse")) %>% 
  fit(train_cleaned)

modeltime_table(wflw_fit_glmnet_tuned) %>% 
  modeltime_calibrate(new_data = testing(splits) %>% 
                        filter(symbol == "LLY")) %>% 
  modeltime_forecast(actual_data = data_prepared_tbl %>% 
                       filter(symbol == "LLY"), 
                     new_data = testing(splits) %>% 
                       filter(symbol == "LLY")
  ) %>% 
  plot_modeltime_forecast()

calibrate_and_plot(wflw_fit_glmnet_tuned, print =F)

gc()

# * Prophet Boost TUNE ----

# ** Tunable Specification

model_spec_prophet_boost_tune <- prophet_boost(
  #changepoint_num    = 25,
  #changepoint_range  = 0.8,
  seasonality_yearly = F, #all seasonalities to F because xgboost does that
  seasonality_weekly = F,
  seasonality_daily  = F,
  mtry              = tune(),
  trees             = 250, # leave so it stabilizes the model
  #min_n             = tune(),
  tree_depth        = tune(),
  learn_rate        = tune(),
  stop_iter         = 30,
  loss_reduction    = 0.0005 #tune()
  ) %>% set_engine("prophet_xgboost",  
                   counts = F, 
                   nthread =  -1, 
                   tree_method = "hist",
                   validation = 0.2)

wflw_spec_prophet_boost_tune <- workflow() %>% 
  add_model(model_spec_prophet_boost_tune) %>% 
  add_recipe(recipe_spec)

# ** Tuning

set.seed(69)
tic()
tune_results_prophet_boost <- wflw_spec_prophet_boost_tune %>% 
  tune_race_anova(
    resamples = resamples_kfold,
    param_info = extract_parameter_set_dials(wflw_spec_prophet_boost_tune) %>% 
      update(learn_rate = learn_rate(range = c(0.1, 0.4), trans = NULL),
             #trees      = trees(range = c(50,1000)),
             mtry       = mtry_prop(range = c(0.1,0.7))
      ),
    grid = 4,
    control = control_race(verbose_elim = T, allow_par = T)
  )
toc()

# ** Results

tune_results_prophet_boost %>% 
  show_best(metric = "rmse", n = Inf)

tune_results_prophet_boost %>% 
    show_best(metric = "rsq", n = Inf)

# ** Finalize

start_pb <- Sys.time()
wflw_fit_prophet_boost_tuned <- wflw_spec_prophet_boost_tune %>% 
  finalize_workflow(select_best(tune_results_prophet_boost, metric = "rmse")) %>% 
  fit(train_cleaned)
end_pb <- Sys.time()
end_pb-start_pb

modeltime_table(wflw_fit_prophet_boost_tuned) %>% 
  modeltime_calibrate(new_data = testing(splits) %>% 
                        filter(symbol == "LLY")) %>% 
  modeltime_forecast(actual_data = data_prepared_tbl %>% 
                       filter(symbol == "LLY"), 
                     new_data = testing(splits) %>% 
                       filter(symbol == "LLY")
  ) %>% 
  plot_modeltime_forecast()

calibrate_and_plot(wflw_fit_prophet_boost_tuned, print = F)

gc()

# * Prophet lg Boost TUNE SKIP ----

# ** Tunable Specification

model_spec_prophet_boost_tune <- boost_prophet( #prophet_boost for modeltime
  #changepoint_num    = 25,
  #changepoint_range  = 0.8,
  seasonality_yearly = F, #all seasonalities to F because xgboost does that
  seasonality_weekly = F,
  seasonality_daily  = F,
  mtry              = tune(),
  trees             = 500, # leave so it stabilizes the model
  #min_n             = tune(),
  tree_depth        = tune(),
  learn_rate        = tune()
  #loss_reduction    = tune()
) %>% set_engine("prophet_lightgbm", counts = F)

wflw_spec_prophet_boost_tune <- workflow() %>% 
  add_model(model_spec_prophet_boost_tune) %>% 
  add_recipe(recipe_spec)

# ** Tuning

set.seed(69)
tic()
tune_results_prophet_boost <- wflw_spec_prophet_boost_tune %>% 
  tune_grid(
    resamples = train_cleaned %>% vfold_cv(v = 2), #resamples_kfold,
    param_info = extract_parameter_set_dials(wflw_spec_prophet_boost_tune) %>% 
      update(learn_rate = learn_rate(range = c(0.1, 0.4), trans = NULL),
             #trees      = trees(range = c(50,1000)),
             mtry       = mtry_prop(range = c(0.1,0.9))
      ),
    grid = 2,
    control = control_grid(verbose = T, allow_par = F)
  )
toc()

# ** Results

tune_results_prophet_boost %>% 
  show_best(metric = "rmse", n = Inf)

# ** Finalize

wflw_fit_prophet_boost_tuned <- wflw_spec_prophet_boost_tune %>% 
  finalize_workflow(select_best(tune_results_prophet_boost, metric = "rmse")) %>% 
  fit(train_cleaned)


# * Modeling explanation ----

extract_fit_engine(wflw_fit_xgboost_tuned) %>%
  xgboost::xgb.importance(model = .) %>%
  as_tibble() %>% #View()
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
  slice_max(Gain, n = 20) %>% 
  mutate(Feature = as_factor(Feature) %>% fct_rev()) %>% 
  ggplot(aes(x = Gain, y = Feature, fill = Feature))+
  geom_col(alpha = 0.7)+
  scale_fill_tq()+
  theme_few()+
  theme(legend.position = "none")+
  ggtitle("LightGBM Feature Importance")

extract_fit_engine(wflw_fit_lgb_tuned) %>%
  # extract_fit_engine(wflw_fit_lightgbm) %>%
  lightgbm::lgb.importance(model = .) %>% 
  as_tibble() %>%
  arrange(desc(Gain)) %>% 
  mutate(rank = row_number()) %>% 
  filter(str_detect(Feature, "sin|cos"))

# compare xgb and lgbm
library(tidyt)
bind_rows(extract_fit_engine(wflw_fit_xgboost_tuned) %>%
              xgboost::xgb.importance(model = .) %>%
              as_tibble() %>% 
              mutate(model = "xgboost") %>% 
              slice_max(Gain, n = 20),
          extract_fit_engine(wflw_fit_lgb_tuned) %>%
              # extract_fit_engine(wflw_fit_lightgbm) %>%
              lightgbm::lgb.importance(model = .) %>% 
              as_tibble()%>% 
              mutate(model = "lightgbm") %>% 
              slice_max(Gain, n = 20)) %>% 
    ggplot(aes(x = Gain,
               y = reorder_within(x = Feature, by = Gain, within = model))
           )+
    geom_col()+
    facet_wrap(.~model)+
    scale_y_reordered()
          

# scale_fill_viridis_d()# SHAP Analysis
df_explain  <- bake( 
  prep(recipe_spec), 
  has_role("predictor"),
  new_data = train_cleaned %>% slice_sample(prop = 0.2)
  #composition = "matrix"
  ) %>% 
  select(-date)

shap_values <- extract_fit_engine(wflw_fit_lgb_tuned) %>% 
  shapviz(X_pred = data.matrix(df_explain), X = df_explain)

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
  sv_importance("no") %>% enframe()

shap_values %>% 
  sv_dependence(shap_names$name[1], alpha = 0.7) +
  scale_color_gradient(high = "#ff0d57", low = "#1e88e5")+
  theme_minimal()

shap_values %>% 
  sv_dependence("Close_macd_signal_12_26_9", alpha = 0.7) +
  scale_color_gradient(high = "#ff0d57", low = "#1e88e5")+
  theme_minimal()

shap_values %>% 
  sv_dependence(shap_names$name[3], alpha = 0.7) +
  scale_color_gradient(high = "#ff0d57", low = "#1e88e5")+
  geom_smooth(se = F,colour = "gray",linewidth = 1)+
  theme_minimal()

shap_values %>% 
  sv_dependence("Close_rsi_21", alpha = 0.7) +
  scale_color_gradient(high = "#ff0d57", low = "#1e88e5")+
  geom_smooth(se = F,colour = "gray",linewidth = 1)+
  theme_minimal()

shap_values %>% 
  sv_dependence("Close_macd_12_26_9", alpha = 0.7) +
  scale_color_gradient(high = "#ff0d57", low = "#1e88e5")+
  #geom_smooth(se = F,colour = "gray",linewidth = 1)+
  theme_minimal()

shap_values %>% 
  sv_dependence("index.num", alpha = 0.7) +
  scale_color_gradient(high = "#ff0d57", low = "#1e88e5")+
  theme_minimal()

xvars <- c("Vol_WAP_norm","Close_fastqsmom_21_252_126","Close_natr_63","Close_rel_vol_14")
evars <- colnames(shap_values$S)[grep("eps", colnames(shap_values$S))]

shap_values %>% 
  sv_dependence(xvars, viridis_args = list(option = "viridis", direction = -1))

shap_values %>% 
  sv_dependence(evars, viridis_args = list(option = "viridis", direction = -1))

# SHAP interactions for fwd return
shap_values %>%  
  sv_dependence("date_sin252_K1", 
                color_var = xvars, 
                interactions = TRUE,  
                viridis_args = list(option = "viridis", direction = -1))

shap_values %>% 
  sv_dependence2D("Vol_WAP_norm","Return_fwd_10", alpha = 0.7) +
  scale_color_gradient(high = "#ff0d57", low = "#1e88e5")+
  theme_minimal()


# 5.0 EVALUATE TUNED FORECASTS  -----
# * Model Table ----
submodels_tbl <- modeltime_table(
    wflw_fit_xgboost_tuned
    ,wflw_fit_glmnet_tuned
    #wflw_fit_earth_tuned,
    ,wflw_fit_lgb_tuned
    ,wflw_fit_prophet_boost_tuned
    #,wflw_fit_rf_tuned
    ) %>% 
  update_model_description(1, "XGBOOST - Tuned") %>% 
  update_model_description(2, "GLMNet - Tuned") %>% 
  update_model_description(3, "LightGBM - Tuned") %>% 
  update_model_description(4, "Prophet Boost - Tuned") #%>% 
  # update_model_description(5, "RandomForest - Tuned") 

# * Calibration ----
calibration_tbl <- submodels_tbl %>% 
    modeltime_calibrate(testing(splits), id = "symbol")

# * Accuracy ----
calibration_tbl %>% 
    modeltime_accuracy(metric_set = extended_forecast_accuracy_metric_set()) %>% 
    arrange(rmse)

calibration_tbl %>% 
  modeltime_accuracy(acc_by_id = T) %>% #View()
  #arrange(rmse) %>% 
  select(.model_desc, rmse:rsq) %>% 
  pivot_longer(-.model_desc) %>% 
  ggplot(aes(x = .model_desc, y = value, fill = .model_desc))+
  geom_boxplot(alpha = 0.7, show.legend = F) +
  facet_grid(name~., scales = "free_y")+
  theme_few()

# * Forecast Test ----
gc()
forecast_test <- calibration_tbl %>% 
    modeltime_forecast(
        new_data   = testing(splits),
        actual_data = data_prepared_tbl %>% drop_na(),
        keep_data  = T) # keeps the grouping variable

forecast_test %>% 
    group_by(symbol) %>%
  filter(.index >= "2023-01-01") %>% 
    plot_modeltime_forecast(
        .facet_ncol = 2,
        .conf_interval_show = F,
        .interactive = T,
        .trelliscope = T
    )

test_fcst <- forecast_test %>% 
  filter(.index == max(.index) & symbol == "NVDA")

# 6.0 RESAMPLING ----
# - Assess the stability of our models over time
# - Helps us strategize an ensemble approach

# * Time Series CV ----

resamples_tscv <- train_cleaned %>% 
    time_series_cv(
        initial     = 252*3.5,
        assess      = 21,
        skip        = 1,
        cumulative  = T,
        slice_limit = 36
    )

resamples_tscv %>% 
    tk_time_series_cv_plan() %>% 
    plot_time_series_cv_plan(date, Return_fwd_21,
                             .facet_ncol = 3,
                             .interactive = F)
  

# * Fitting Resamples ----

model_tbl_tuned_resamples <- submodels_2_tbl %>% 
    modeltime_fit_resamples(
        resamples = resamples_tscv,
        control   = control_resamples(verbose = T, allow_par = T)
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

ensemble_fit <- submodels_tbl %>% 
    #filter(.model_id %in% submodels_ids_to_keep) %>% 
    #ensemble_average()
    ensemble_average(type = 'median') # reduces effects of bad forecasts and reduces overfitting

model_ensemble_tbl <- modeltime_table(
    ensemble_fit
)

# * Accuracy ----
model_ensemble_tbl %>% 
  modeltime_accuracy(testing(splits), metric_set = extended_forecast_accuracy_metric_set())

# * Weighted ensemble ----
loadings_tbl <- submodels_tbl %>% 
  #filter(.model_id %in% submodels_ids_to_keep) %>% 
  modeltime_accuracy(testing(splits)) %>%
  mutate(rank = min_rank(-rmse)) %>%
  select(.model_id, rank)

ensemble_fit_wt <- submodels_tbl %>%
  ensemble_weighted(loadings = loadings_tbl$rank)

ensemble_fit_wt$fit$loadings_tbl

model_ensemble_tbl_wt <- modeltime_table(
  ensemble_fit_wt
) 

model_ensemble_tbl_wt %>%
  modeltime_accuracy(testing(splits), 
                     metric_set = extended_forecast_accuracy_metric_set())

# * lm stack ----

# refit samples
submodels_resamples_kfold_tbl <- submodels_tbl %>% 
 # filter(.model_id %in% submodels_2_ids_to_keep) %>% 
  modeltime_fit_resamples(
    resamples = resamples_kfold,
    control   = control_resamples(
      verbose   = T,
      allow_par = T
    )
  )

prophet_boost_resamples_tbl <- modeltime_table(
  wflw_fit_prophet_boost_tuned
  ) %>%  
  modeltime_fit_resamples(
    resamples = resamples_kfold,
    control   = control_resamples(
      verbose   = T,
      allow_par = T
      )
    )

submodels_resamples_kfold_tbl_c <- combine_modeltime_tables(submodels_resamples_kfold_tbl %>% 
                                                              filter(.model_id %in% c(1:3)), 
                                                            prophet_boost_resamples_tbl)

# fit lm
ensemble_fit_lm <- submodels_resamples_kfold_tbl %>% 
  ensemble_model_spec(
    model_spec = linear_reg() %>% set_engine("lm"),
    control = control_grid(verbose = T)
  )

# test accuracy
modeltime_table(
  ensemble_fit_lm
) %>% 
  modeltime_accuracy(testing(splits))

# 8.0 Ensemble Forecast test ----
gc()
forecast_ensemble_test_tbl <- model_ensemble_tbl_wt %>% 
    modeltime_forecast(
        new_data = testing(splits),
        actual_data = data_prepared_tbl,
        keep_data = T
    )

forecast_ensemble_test_tbl %>% 
    group_by(symbol) %>% 
    plot_modeltime_forecast(
        .facet_ncol = 2,
        .conf_interval_show = F,
        .interactive = T,
        .legend_show = F,
        .trelliscope = T
    )

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
data_prepared_clean_tbl <- data_prepared_tbl %>% 
  filter_by_time(date, .start = max(date)-years(3)) %>% 
  # group_by(symbol) %>% 
  #mutate(Return_fwd_21 = ts_clean_vec(Return_fwd_21, period = 7)) %>% 
  drop_na()

model_ensemble_refit_tbl <- model_ensemble_tbl_wt %>% 
    modeltime_refit(data_prepared_clean_tbl)

# * Ensemble final forecast ----
model_ensemble_final_forecast <- model_ensemble_refit_tbl %>% 
    modeltime_forecast(
        new_data    = forecast_tbl,
        actual_data = data_prepared_clean_tbl,
        keep_data = T
    )  

model_ensemble_final_forecast %>% 
  arrange(desc(date)) %>% 
    group_by(symbol) %>% 
    plot_modeltime_forecast(
        .facet_ncol = 2,
        .y_intercept = 0,
        .conf_interval_show = F,
        .legend_show = F,
        .trelliscope = T
    )

model_ensemble_final_forecast %>% 
  filter(date == max(date)) %>% 
  select(symbol, .value, date, Vol_WAP_norm) %>% 
  arrange(desc(.value)) %>% 
  print(n = nrow(.))

# * Turn OFF Parallel Backend ----
plan(sequential)

# save forecasts ----
write_rds(model_ensemble_final_forecast,
          str_glue("04-Financial/04_01_save_data/{today()}_model_ensemble_final_forecast.rds"))
