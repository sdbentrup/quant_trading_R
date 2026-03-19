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
                actual_data = data_prepared_tbl
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
sp400 <- tq_index("SP400")
# sp600 <- tq_index("SP600")

exclude_symbols <- c("TSLA","PLTR")

sp500_symbols <- sp500 %>% 
    filter(symbol != "-" & !str_detect(company,"CL C")) %>% 
    filter(symbol %notin% exclude_symbols) %>% 
    slice_max(weight, n = 350) %>%
    # slice_sample(prop = 0.1) %>%
    arrange(symbol) %>% 
    pull(symbol) 

sp400_symbols <- sp400 %>% 
    filter(symbol != "-" & !str_detect(company,"CL C")) %>% 
    filter(symbol %notin% exclude_symbols) %>% 
    slice_max(weight, n = 50) %>%
    # slice_sample(prop = 0.1) %>%
    arrange(symbol) %>% 
    pull(symbol) 

symbols <- union(sp500_symbols, sp400_symbols)

# * get stock price data ----
prices_base <- tq_get(symbols, from = from) %>% 
    setDT()

# ** view data ----
prices_base

prices_base[,.(.N),keyby = symbol] %>% arrange(N)
prices_base[,(uniqueN(symbol))]

# * reduce data with filters ----
# remove cases with less than 4 years of data
symbols_filtered_list <- prices_base[!is.na(close),
                                     .(count = .N,
                                       max = max(date),
                                       min = min(date)),
                                     keyby = symbol][count >= 252*2.5 & max == max(max),
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

# Engineered indicators ----
# https://blog.elearnmarkets.com/best-25-technical-indicators/

# * function for features ----
add_features <- function(prices_dt, price) {
  price_col <- prices_dt[[price]]
  
  # Calculate Indicators
  prices_dt[, Close_macd_long := MACD(price_col, 50, 200, 30)[, "macd"]]
  prices_dt[, Close_macd_long_signal := MACD(price_col, 50, 200, 30)[, "signal"]]
  prices_dt[, Close_macd_short := MACD(price_col, 12, 26, 9, list(list(EMA, wilder=TRUE),list(EMA, wilder=TRUE),list(EMA, wilder=TRUE)))[, "macd"]]
  prices_dt[, Close_macd_short_signal := MACD(price_col, 12, 26, 9, list(list(EMA, wilder=TRUE),list(EMA, wilder=TRUE),list(EMA, wilder=TRUE)))[, "signal"]]
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
  prices_dt[, Close_oscillator_7_14_28 := ultimateOscillator(prices_dt[, .(high, low, price_col)])]
  prices_dt[, Close_roc_0_1      := ROC(price_col, n = 1)]
  #prices_dt[, Close_roc_0_1_roll := frollmean(Close_roc_0_1, n=21, align = "right", na.rm = T)]
  prices_dt[, Close_roc_0_5      := ROC(price_col, n = 5)]
  prices_dt[, Close_roc_0_21     := ROC(price_col, n = 21)]
  prices_dt[, Close_roc_0_1_rolling_std_win_63 := frollsd(Close_roc_0_1, 63, fill = NA, align = "right")]
  prices_dt[, Close_natr_63      := ATR(prices_dt[,.(high, low, price_col)], n = 63)[,"atr"]/eval(quote(get(price)))]
  # prices_dt[, Close_rsi_10       := RSI(price_col, n = 10)]
  prices_dt[, Close_rsi_28       := RSI(price_col, n = 28)]
  prices_dt[, Close_cmo_28       := CMO(price_col, n = 28)]
  prices_dt[, Close_cmo_ma       := EMA(Close_cmo_28, n = 21)] # added 6 Mar 26
  prices_dt[, Close_cmo_signal   := Close_cmo_28/Close_cmo_ma] # added 6 Mar 26
  prices_dt[, Close_rolling_mean_126_norm := frollmean(price_col, 126) / price_col] #used to be standardize vec
  prices_dt[, Close_rolling_std_126  := frollsd(price_col, 126, align = "right")]
  
  prices_dt[, Close_SNR_21       := SNR(prices_dt[, .(high, low, price_col)], n = 21)]
  prices_dt[, Close_rel_vol_14   := 100 - 100 / (1 + frollsd(price_col, 14))]
  prices_dt[, Close_252_max_diff := price_col/frollmax(price_col, 252, align = 'right')] # modified from frollapply FUN = max
  prices_dt[, Close_252_min_diff := price_col/frollmin(price_col, 252, align = 'right')] # modified from frollapply FUN = min
  prices_dt[, Close_kst          := KST(price_col)[,"kst"]]
  prices_dt[, Close_kst_signal   := KST(price_col)[,"signal"]]
  prices_dt[, Close_TDI          := TDI(price_col, n = 20, multiple = 2)[,"tdi"]]
  prices_dt[, Close_TDI_di       := TDI(price_col, n = 20, multiple = 2)[,"di"]]
  prices_dt[, Close_TRIX         := TRIX(close, n = 20, nSig = 9, "EMA", percent = TRUE)[,"TRIX"]]
  prices_dt[, Close_TRIX_signal  := TRIX(close, n = 20, nSig = 9, "EMA", percent = TRUE)[,"signal"]]
  prices_dt[, SAR                := SAR(prices_dt[, .(high, low)])]
  prices_dt[, SMI                := SMI(prices_dt[, .(high, low, price_col)])[,"SMI"]]
  prices_dt[, SMI_signal         := SMI(prices_dt[, .(high, low, price_col)])[,"signal"]]
  prices_dt[, CMF                := CMF(prices_dt[, .(high, low, price_col)], volume)]
  prices_dt[, EMV                := EMV(prices_dt[, .(high, low)], volume)[,"emv"]] # added 6 Mar 26
  prices_dt[, maEMV              := EMV(prices_dt[, .(high, low)], volume)[,"maEMV"]] # added 6 Mar 26
  
  prices_dt[, Close_ADX_DIp      := ADX(prices_dt[,.(high, low, price_col)])[,"DIp"]]
  prices_dt[, Close_ADX_DIn      := ADX(prices_dt[,.(high, low, price_col)])[,"DIn"]]
  prices_dt[, Close_ADX_DX       := ADX(prices_dt[,.(high, low, price_col)])[,"DX"]] # reactivated 6 Mar 26
  prices_dt[, Close_ADX_ADX      := ADX(prices_dt[,.(high, low, price_col)])[,"ADX"]]
  
  # relative vigor index # added 6 Mar 2026
  prices_dt[,":=" (numSMA = SMA(price_col - open, n = 10), denSMA = SMA(high - low, n = 10))][,":=" (rvi = numSMA/denSMA, rvi_signal = SMA(numSMA/denSMA, 4))]
  prices_dt[,":=" (numSMA = NULL, denSMA = NULL)]
  
  # create indicators for intraday effects and day return
  # turn these off since they are low importance
  # prices_dt[, ":=" (intraday = high/low-1, day_return = open/close-1)]
  
  # Volume-based indicators
  # prices_dt[, Vol_ema_21_norm  := EMA(volume, n = 21) / volume]
  prices_dt[, Vol_roc_0_1      := ROC(volume, n = 1)]
  
  # create indicators based on VWAP
  prices_dt[, Vol_WAP           := VWAP(price_col, volume)] # only used for ratios, then dropped
  # prices_dt[, Vol_WAP_norm      := standardize_vec(Vol_WAP)]
  prices_dt[, Vol_WAP_Close     := Vol_WAP/price_col]
  prices_dt[, Vol_WAP_EMA       := EMA(Vol_WAP, n = 21)] # only used for ratios, then dropped
  # prices_dt[, Vol_WAP_EMA_norm  := standardize_vec(EMA(Vol_WAP, n = 21))]
  prices_dt[, Vol_WAP_EMA_ratio := Vol_WAP/Vol_WAP_EMA]
  prices_dt[, Vol_WAP_ROC       := ROC(Vol_WAP, n = 21)]
  prices_dt[, Vol_roc_0_1_rolling_std_63 := frollsd(Vol_roc_0_1, 63, fill = NA, align = "right")]
  
  # standardize VWAP by 252 days
  prices_dt[, Vol_WAP_norm      := (Vol_WAP - frollmean(Vol_WAP, n = 252, align = "right"))/frollsd(Vol_WAP, n = 252, align = "right")]
  # prices_dt[, Vol_WAP_roll_ratio := Vol_WAP/Vol_WAP_norm_roll]
  
  prices_dt[, ':=' (Vol_WAP = NULL, 
                    Vol_WAP_EMA = NULL
                    )]
  
  # prices_dt[, Vol_OBV_norm := standardize_vec(OBV(price_col, volume))]
  prices_dt[, Vol_OBV     := OBV(price_col, volume)]
  prices_dt[, Vol_OBV_std := (Vol_OBV - frollmean(Vol_OBV, n = 126, align = 'right'))/frollsd(Vol_OBV, n = 126, align = 'right')]
  prices_dt[, Vol_OBV     := NULL]
  
  prices_dt[, Vol_MFI_21 := MFI(prices_dt[,.(high, low, close)], volume, n = 21)]

  # Momentum
  prices_dt[, `:=`(
    Close_lag_1   = data.table::shift(price_col, n = 1),
    Close_lag_21  = data.table::shift(price_col, n = 21),
    Close_lag_252 = data.table::shift(price_col, n = 252)
  )]
  
  prices_dt[, Close_momentum_21_252_126 := ((Close_lag_21 / Close_lag_252 - 1) - (Close_lag_1 / Close_lag_21 - 1))/frollsd(Close_roc_0_1, 126, fill = NA, align = "right")] # revised 6 Mar 26 to match Python version
  
  # Z-score features
  prices_dt[, Close_zscore_126 := (price_col - frollmean(close, 126, align = "right")) / Close_rolling_std_126]
  
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

# Dividends ----
# * function for dividends for multiple symbols ----
dividends_tbl <- map(symbols_filtered_list, 
                     ~ getDividends(.x, 
                                    from = floor_date(from, 'quarter')) |> 
                         fortify.zoo(melt  = TRUE, 
                                     names = c("date", "symbol", "dividend")) |>
                         mutate(symbol = str_remove(symbol,".div"))) %>% 
    list_rbind()

setDT(dividends_tbl)

# setnames(dividends_tbl, "name","symbol")

setorderv(dividends_tbl, cols = c("symbol", "date"))

# modify dates tp prevent dividends on a weekend
dividends_tbl[, date := fcase(wday(date) == 7, date + days(2),
                              wday(date) == 1, date + days(1),
                              default = date)]

dividends_tbl[, div_roc := ROC(dividend,1), keyby = symbol]
dividends_tbl[, number_dividends := 1:.N, keyby = symbol]

# * merge dividends to prices ----
prices_div_dt <- merge(
    prices_features_dt,
    dividends_tbl,
    by = c("symbol","date"),
    all.x = T
)

prices_div_dt[, c("dividend",'div_roc',"number_dividends") := lapply(.SD, function(x) nafill(x, "locf")), 
              .SDcols = c("dividend",'div_roc',"number_dividends"), 
              by = symbol]

prices_div_dt[,":=" (div_ann_days = round(as.numeric(difftime(date,
                                                              min(date),
                                                              units = "days")),0)),
              keyby = .(symbol, number_dividends)]

#prices_div_dt[, dividend_std := standardize_vec(dividend), keyby = symbol]

prices_div_dt[, div_close_ratio := dividend / close]

# * fill dividends ----
setnafill(prices_div_dt,  type = 'const', fill = 0, cols = colnames(prices_div_dt[,.SD, .SDcols = patterns("div")]))

# Earnings data ----

# * function for earnings for multiple symbols ----

get_earnings_data <- function(symbol){
    
    #symbol <- gsub(x=symbol, pattern="\\-", replacement="\\.")
    
    symbol <- gsub("[/-]", ".", symbol)
    
    
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
    # use absolutes to avoid divide by 0 and other issues
    # eps_surprise shows how much higher or lower the eps was than the estimate
    # eps growth is the change in actual EPS between announcements
    # ann_number splits the announcements this is used later for calculating the days between announcements
    earnings_table[,":=" (eps_surprise = act_eps-est_eps,
                          eps_growth = act_eps-data.table::shift(act_eps, 1, type = "lag"),
                          #announcement = "Y",
                          num_ann = 1:.N)]
    
    earnings_table <- earnings_table[,select(.SD, -(`Announcement Date`:`Actual EPS`))]
    earnings_table[, symbol := gsub("\\.", "-", symbol)]
}

earning_symbols <- unique(prices_features_dt$symbol)

# Use lapply to apply the function to each symbol
earnings_data_list <- future_lapply(earning_symbols,
                                    get_earnings_data)

# Combine the results into a single data.table
earnings_data <- rbindlist(earnings_data_list, fill = TRUE)

setkey(earnings_data, symbol, date)

# * merge eps to prices ----

# this creates NAs where there were no earnings
# we will fill these in later

prices_earnings_tbl <- merge(prices_div_dt,
                             earnings_data,
                             by = c("date","symbol"),
                             all.x = T)

setorderv(prices_earnings_tbl, cols = c("symbol","date"))

# * fill earnings ----
# discount the earnings later by the time with an interaction term
fill_cols <- patterns("eps|num_ann", cols = colnames(prices_earnings_tbl))

prices_earnings_tbl[, (fill_cols) := lapply(.SD, function(x) nafill(x, "locf")), 
                    .SDcols = fill_cols, 
                    by = symbol]

setnafill(prices_earnings_tbl, type = "const", 0, cols = "num_ann")

# this is used later as an interaction variable to see if the time since the announcement affects prediction
prices_earnings_tbl[,":=" (eps_ann_days = round(as.numeric(difftime(date,
                                                               min(date),
                                                               units = "days")),0),
                           pe_ratio = act_eps/close),
                    keyby = .(symbol, num_ann)]

summary(prices_earnings_tbl)

# ** test on eps ----
prices_earnings_tbl %>% 
    filter(!is.na(est_eps))

prices_earnings_tbl %>% 
    filter(!is.na(est_eps) & symbol == "NVDA" )

prices_earnings_tbl %>% 
    filter(is.na(est_eps))

prices_earnings_tbl %>% 
    filter(symbol == "WBA" & date == "2015-09-08")

# Commodities ----
commodities <- c(
    "CL=F" # crude oil futures
    ,"GC=F" #gold futures
)
commodity_prices <- tq_get(commodities, from = from-30)

setDT(commodity_prices)
setorderv(commodity_prices, c("symbol","date"))
setnafill(commodity_prices, type = "locf", cols = colnames(commodity_prices[,.SD,.SDcols = sapply(commodity_prices, is.numeric)]))

commodity_prices[, symbol := str_replace_all(symbol,
                                             c("CL=F" = "crude",
                                               "GC=F" = "gold"))]

commodity_prices[,":=" (close_roc_5 = close/data.table::shift(close, 5)-1,
                        close_roc_21 = close/data.table::shift(close, 21)-1,
                        close_max = close/frollmax(close, 252, align = "right"),
                        close_min = close/frollmin(close, 252, align = "right"),
                        close_norm = (close - frollmean(close, 252))/frollsd(close,252)),
                 keyby = .(symbol)]

commodity_prices[, ":=" (
    close_macd = EMA(close, 12) - EMA(close, 26)
), 
keyby = symbol][,":=" (close_macd_signal = EMA(close_macd, 9)), 
                keyby = symbol]

cast_cols <- colnames(commodity_prices[,.SD, .SDcols = patterns("close")])

commodity_prices_cast <- commodity_prices[,dcast.data.table(.SD, date ~ symbol, value.var = cast_cols)]

# * merge to prices data ----
prices_comm_tbl <- merge(prices_earnings_tbl, 
                         commodity_prices_cast, 
                         all.x = T,
                         by = "date")

prices_comm_tbl[,":=" (Close_ratio_crude = close/close_crude,
                       Close_ratio_gold  = close/close_gold)]

# Macroeconomic data ----
# * load macroeconomic data ----

# start the from data 3 months earlier to try to get an announcement before
# the start of the prices data

indicators <- c("GDPC1",    # real gdp
                "CPILFESL", # core cpi # https://fred.stlouisfed.org/series/CPILFESL
                "UNRATE",   # unemployment rate
                "FEDFUNDS", # fed funds effective rate
                "UMCSENT",  # u michigan consumer sentiment
                "M2SL"      # M2 money supply
                )

economic_data <- tq_get(indicators, get = "economic.data", from = from-days(90))
setDT(economic_data)

economic_data[,.N, keyby = symbol]

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
                                            #value_FEDFUNDS,
                                            value_UNRATE,
                                            value_UMCSENT,
                                            change_UNRATE,
                                            change_CPILFESL, 
                                            change_GDPC1,
                                            change_M2SL,
                                            change_UMCSENT,
                                            econ_ann_days)]

#economic_data_final[,unrate_over := fifelse(value_UNRATE > frollmean(value_UNRATE, n = 300),1,0)]

# * join economic data to prices ----
prices_earnings_econ_tbl <- merge(prices_comm_tbl, 
                                  economic_data_final, 
                                  all.x = T,
                                  by = "date")

# Create full data ----
# * full data ----
full_data <- prices_earnings_econ_tbl %>% 
    
    filter(!is.na(Close_momentum_21_252_126)) %>% # filter NA on momentum since it is the longest indicator
    
    # remove unneeded features
    select(-(open:adjusted),
           #-Return_fwd_5, -Return_fwd_10,
           -num_ann,
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
         act_eps_norm = standardize_vec(act_eps),
         eps_surprise_norm = standardize_vec(eps_surprise),
         eps_growth_norm = standardize_vec(eps_growth)) %>% 
  ungroup() %>% 
  select(-est_eps, -act_eps, -eps_surprise, -eps_growth) %>% 
  mutate(symbol = as_factor(symbol)) %>% 
  
  # lags / rolling / fourier features # fourier since it is panel data, different features by series
  group_by(symbol) %>% 
  arrange(symbol,date) %>% 
  
  # removing the fourier features since these do not really add to accuracy or shap values
  tk_augment_fourier(date, .periods = c(63,252), .K = 1) %>%
    ungroup() %>% 
    rowid_to_column(var = 'rowid') %>% 
  setDT()

setorderv(full_data,cols = c("date","symbol"))

# Save data for future use ----
write_rds(full_data, str_glue("01_save_data/{today()}_full_data.rds"))

# * remove unneeded data from environment ----
rm(prices_features_list)
rm(earnings_data)
rm(earnings_data_list)
rm(economic_data_cast)
rm(economic_data_pad)
rm(economic_data_final)
rm(prices_earnings_tbl)
rm(prices_earnings_econ_tbl)
rm(prices_features_dt)
rm(prices_comm_tbl)
rm(prices_div_dt)
rm(prices_base)
rm(dividends_tbl)
rm(economic_data)
rm(commodity_prices)
rm(commodity_prices_cast)
# not rm(prices_dt) since we can use this for the optimization
rm(list=ls(pattern="symbol"))
gc()

#rm(list=ls(pattern="^wflw_"))

# 2.0 TIME TEST/TRAIN SPLIT ----
# * clean for splitting ----
# remove NA values, add fourier transform features, set order
data_prepared_tbl <- full_data %>% 
    filter(!is.na(Return_fwd_21))

# * Split into train and forecast sets ----

forecast_tbl <- full_data %>% 
  filter(is.na(Return_fwd_21))

# setnafill(data_prepared_tbl, type = "const", fill = 0, cols = "eps_surprise_norm")
    
splits <- data_prepared_tbl %>% 
    time_series_split(
    date_var = date,
    initial = 252 * 2,#"2 years",
    assess  = "45 days",
    cumulative = F
    )

# 3.0 RECIPE ----

# * Recipe Specification ----

recipe_spec <- recipe(Return_fwd_21 ~ ., data = training(splits)) %>%
    update_role(rowid, new_role = 'identifier') %>% 
    update_role(symbol, new_role = 'symbol') %>%
    # step_mutate_at(symbol, fn = droplevels) %>%
    # step_timeseries_signature(date) %>%
    # step_rm(matches("(.xts$)|(.iso$)|(.hour)|(.minute)|(.second)|(.am.pm)")) %>%
    # step_rm(date_index.num, date_year) %>%
    # step_normalize(date_index.num, ends_with("_year")) %>%  
    # step_normalize(index.num,year) %>% 
    step_rm(Return_fwd_5, Return_fwd_10) %>%
    # step_rm(number_dividends) %>% 
    # step_dummy(all_nominal_predictors(), one_hot = T, keep_original_cols = F) %>%
    # step_mutate(spy_state = as.factor(spy_state)) %>% 
    # step_interact(~Close_macd_long:Close_macd_short) %>% 
    step_interact(~Close_macd_long_signal:Close_macd_short_signal) %>% 
    # step_interact(~div_roc:div_ann_days) %>%
    step_interact(~eps_growth_norm:eps_ann_days) %>%
    step_interact(~eps_surprise_norm:eps_ann_days) %>%
    # step_interact(~value_FEDFUNDS:econ_ann_days) %>%
    # step_interact(~value_UNRATE:econ_ann_days) %>%
    # step_interact(~change_CPILFESL:econ_ann_days) %>%
    step_interact(~change_GDPC1:econ_ann_days) %>% 
    step_rm(econ_ann_days, change_GDPC1) %>% 
    step_filter_missing(all_predictors(), threshold = 0.2) %>% 
    step_zv(all_predictors()) %>% 
    step_nzv(all_predictors(), unique_cut = 0.02)


# 4.0 HYPERPARAMETER TUNING MODELS ---- 

# * RESAMPLES - K-FOLD ----- 
set.seed(69)
resamples_kfold <- training(splits) %>% vfold_cv(v = 5)

# * Parallel Processing ----
# cl <- parallel::makeCluster(2, timeout = 60)
# plan(cluster, workers = cl)

options(future.globals.maxSize = 2.3 * 1024^3)  # Set to 2.0 GiB

parallel_start(1:4, .method = "future")

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
    resamples = resamples_kfold,
    param_info = extract_parameter_set_dials(wflw_spec_lgb_tune) %>% 
      update(learn_rate = learn_rate(range = c(0.05, 0.5), trans = NULL),
             trees      = trees(range = c(200,4000)),
             mtry       = mtry_prop(range = c(0.1,0.8))
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

# * accuracy on testing ----
augment(wflw_fit_lgb_tuned,testing(splits)) %>% 
    mutate(error = .pred-Return_fwd_21) %>% 
    summarise(rmse = sqrt(mean(error^2)))

augment(wflw_fit_lgb_tuned,testing(splits)) %>% 
    rsq(.pred, Return_fwd_21)

testing_symbol <- 'AAPL'

fcst_test_fit_lgb_tuned <- modeltime_table(wflw_fit_lgb_tuned) %>% 
    modeltime_calibrate(new_data = testing(splits) %>% 
                            filter(symbol == testing_symbol)) %>% 
    modeltime_forecast(actual_data = data_prepared_tbl %>% 
                           filter(symbol == testing_symbol), 
                       new_data = testing(splits) %>% 
                           filter(symbol == testing_symbol)
    )

fcst_test_fit_lgb_tuned %>%
    select(-.model_id,-.model_desc, -.conf_lo, -.conf_hi) %>% 
    #filter(.index == "2025-09-17") %>% 
    pivot_wider(names_from = .key, values_from = .value) %>% 
    filter(!is.na(prediction)) %>% 
    mutate(direction = if_else((actual >= 0 & prediction >= 0)|
                                   (actual < 0 & prediction < 0),
                               1, 0)) %>% 
    summarise(sum = sum(direction), mean = mean(direction))

fcst_test_fit_lgb_tuned %>% 
  plot_modeltime_forecast(.conf_interval_show = T)

# ** save tune results ----
write_rds(tune_results_lgb, "02_models/tune_results_lgb.rds")
rm(tune_results_lgb)
rm(wflw_spec_lgb_tune) # can recreate spec easily

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
        resamples = resamples_kfold,
        param_info = extract_parameter_set_dials(wflw_spec_xgboost_tune) %>% 
          update(learn_rate = learn_rate(range = c(0.05, 0.5), trans = NULL),
                 trees      = trees(range = c(200,4000)),
                 mtry       = mtry_prop(range = c(0.1,0.8))
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

# ** test forecast ----
augment(wflw_fit_xgboost_tuned,testing(splits)) %>% 
    mutate(error = .pred-Return_fwd_21) %>% 
    summarise(rmse = sqrt(mean(error^2)))

augment(wflw_fit_xgboost_tuned,testing(splits)) %>% 
    rsq(.pred, Return_fwd_21)

fcst_test_fit_xgboost_tuned <- modeltime_table(wflw_fit_xgboost_tuned) %>% 
  modeltime_calibrate(new_data = testing(splits) %>% 
                        filter(symbol == testing_symbol)) %>% 
  modeltime_forecast(actual_data = data_prepared_tbl %>% 
                       filter(symbol == testing_symbol), 
                     new_data = testing(splits) %>% 
                       filter(symbol == testing_symbol)
  ) 

fcst_test_fit_xgboost_tuned %>% 
  plot_modeltime_forecast()

fcst_test_fit_xgboost_tuned %>%
    select(-.model_id,-.model_desc, -.conf_lo, -.conf_hi) %>% 
    #filter(.index == "2025-09-17") %>% 
    pivot_wider(names_from = .key, values_from = .value) %>% 
    filter(!is.na(prediction)) %>% 
    mutate(direction = if_else((actual >= 0 & prediction >= 0)|
                                   (actual < 0 & prediction < 0),
                               1, 0)) %>% 
    summarise(sum = sum(direction), mean = mean(direction))


calibrate_and_plot(wflw_fit_xgboost_tuned, plot = F)

# ** save tune results ----
write_rds(tune_results_xgboost, "02_models/tune_results_xgboost.rds", compress = "gz")
rm(tune_results_xgboost)
rm(wflw_spec_xgboost_tune) # remove spec to save memory

gc()

# * Prophet XGBoost TUNE ----

# ** Tunable Specification

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
    add_recipe(recipe_spec)# %>% step_corr(all_numeric_predictors(), threshold = 0.95))

# ** Tuning

set.seed(69)
start <- Sys.time()
tune_results_prophet_boost <- wflw_spec_prophet_boost_tune %>% 
    tune_race_anova(
        resamples = resamples_kfold,
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

# test results on final
augment(wflw_fit_prophet_boost_tuned,testing(splits)) %>% 
    mutate(error = .pred-Return_fwd_21) %>% 
    summarise(rmse = sqrt(mean(error^2)))

augment(wflw_fit_prophet_boost_tuned,testing(splits)) %>% 
    rsq(.pred, Return_fwd_21)

fcst_test_fit_prophet_boost_tuned <- modeltime_table(wflw_fit_prophet_boost_tuned) %>% 
    modeltime_calibrate(new_data = testing(splits) %>% 
                            filter(symbol == testing_symbol)) %>% 
    modeltime_forecast(actual_data = data_prepared_tbl %>% 
                           filter(symbol == testing_symbol), 
                       new_data = testing(splits) %>% 
                           filter(symbol == testing_symbol)
    ) 

fcst_test_fit_prophet_boost_tuned %>% 
    plot_modeltime_forecast()

fcst_test_fit_prophet_boost_tuned %>%
    select(-.model_id,-.model_desc, -.conf_lo, -.conf_hi) %>% 
    #filter(.index == "2025-09-17") %>% 
    pivot_wider(names_from = .key, values_from = .value) %>% 
    filter(!is.na(prediction)) %>% 
    mutate(direction = if_else((actual >= 0 & prediction >= 0)|
                                   (actual < 0 & prediction < 0),
                               1, 0)) %>% 
    summarise(sum = sum(direction), mean = mean(direction))

calibrate_and_plot(wflw_fit_prophet_boost_tuned, plot = F)

# save tune results
write_rds(tune_results_prophet_boost, "02_models/tune_results_prophet_boost.rds")
rm(tune_results_prophet_boost)
rm(wflw_spec_prophet_boost_tune) # remove spec to save memory

gc()

# * glmnet TUNE ----

# ** Tunable Specification

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
  tune_grid(
    resamples = resamples_kfold,
    grid = 6,
    control = control_grid(verbose = T, parallel_over = NULL)
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

modeltime_table(wflw_fit_glmnet_tuned) %>% 
  modeltime_calibrate(new_data = testing(splits) %>% 
                        filter(symbol == testing_symbol)) %>% 
  modeltime_forecast(actual_data = data_prepared_tbl %>% 
                       filter(symbol == testing_symbol), 
                     new_data = testing(splits) %>% 
                       filter(symbol == testing_symbol)
  ) %>% 
  plot_modeltime_forecast()

calibrate_and_plot(wflw_fit_glmnet_tuned, plot =F)

# save tune results
write_rds(tune_results_glmnet, "02_models/tune_results_glmnet.rds")
rm(tune_results_glmnet)
rm(wflw_spec_glmnet_tune) # remove spec to save memory

gc()


# * CatBoost ----
# https://catboost.ai/docs/en/
# https://bonsai.tidymodels.org/
# https://bonsai.tidymodels.org/reference/train_catboost.html

model_spec_catboost <- boost_tree("regression") %>% 
    set_engine('catboost', 
               boosting_type = "Plain",
               early_stopping_rounds = 30)

model_spec_catboost <- boost_tree("regression",
                                  trees  = 2500
                                  #,tree_depth = 5
                                  #,min_n = 20
                                  #,mtry = 5
                                #,stop_iter = 20
                                ) %>% 
    set_engine('catboost',
               early_stopping_rounds = 30, 
               boosting_type = "Plain") 

set.seed(69)
start <- Sys.time()
wflw_fit_cb <- workflow() %>% 
    add_model(model_spec_catboost) %>% 
    add_recipe(recipe_spec %>% step_rm(date)) %>% 
    fit(training(splits))
    #fit_resamples(resamples_kfold)
end <- Sys.time()
end-start

wflw_fit_cb
# collect_metrics(wflw_fit_cb)

extract_fit_engine(wflw_fit_cb) %>% 
    catboost::catboost.get_feature_importance(model = .) %>%
    as_tibble(rownames = 'Variable') %>% 
    #enframe() %>%
    arrange(desc(V1))

# ** accuracy on testing ----

augment(wflw_fit_cb,training(splits)) %>% 
    mutate(error = .pred-Return_fwd_21) %>% 
    summarise(rmse = sqrt(mean(error^2)))

augment(wflw_fit_cb,training(splits)) %>% 
    rsq(.pred, Return_fwd_21)

augment(wflw_fit_cb,testing(splits)) %>% 
    mutate(error = .pred-Return_fwd_21) %>% 
    summarise(rmse = sqrt(mean(error^2)))

augment(wflw_fit_cb,testing(splits)) %>% 
    rsq(.pred, Return_fwd_21)

calibrate_and_plot(wflw_fit_cb, plot = F)

fcst_test_fit_catboost_tuned <- #modeltime_table(wflw_fit_catboost_tuned) %>% 
    modeltime_table(wflw_fit_cb) %>%
    modeltime_calibrate(new_data = testing(splits) %>% 
                            filter(symbol == testing_symbol)) %>% 
    modeltime_forecast(actual_data = data_prepared_tbl %>% 
                           filter(symbol == testing_symbol), 
                       new_data = testing(splits) %>% 
                           filter(symbol == testing_symbol)
    )

fcst_test_fit_catboost_tuned %>%
    select(-.model_id,-.model_desc, -.conf_lo, -.conf_hi) %>% 
    #filter(.index == "2025-09-17") %>% 
    pivot_wider(names_from = .key, values_from = .value) %>% 
    filter(!is.na(prediction)) %>% 
    mutate(direction = if_else((actual >= 0 & prediction >= 0)|
                                   (actual < 0 & prediction < 0),
                               1, 0)) %>% 
    summarise(sum = sum(direction), mean = mean(direction))

fcst_test_fit_catboost_tuned %>% 
    plot_modeltime_forecast(.conf_interval_show = T)

# save tune results
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
              arrange(desc(Gain)) %>% 
              mutate(rank = row_number(),
                     model = "xgboost"),
          extract_fit_engine(wflw_fit_lgb_tuned) %>%
              # extract_fit_engine(wflw_fit_lightgbm) %>%
              lightgbm::lgb.importance(model = .) %>% 
              as_tibble() %>% 
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
    modeltime_forecast(actual_data = data_prepared_tbl %>% 
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
    ,wflw_fit_cb
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
        actual_data = data_prepared_tbl %>% drop_na(),
        keep_data  = T, # keeps the grouping variable and base data
        conf_by_id = T) 

# * backtesting ----
back_fcst <- calibration_tbl %>% 
    modeltime_forecast(
        new_data = data_prepared_tbl %>% 
            filter(date %between% c("2024-12-01","2024-12-10")) %>% 
            drop_na(),
        actual_data = data_prepared_tbl %>% 
            filter(date %between% c("2024-11-01","2024-12-10")) %>% 
            drop_na(),
        keep_data = T,
        conf_by_id = T
    ) 

back_fcst %>% 
    filter(date == "2024-12-06" & .model_desc != "ACTUAL") %>% 
    arrange(desc(.value))

back_fcst %>% 
    filter(date >= "2024-12-01") %>% 
    select(symbol, .value, .model_desc, date) %>% 
    pivot_wider(names_from = .model_desc, values_from = .value, 
                names_repair = "universal") %>%
    group_by(symbol, date) %>% 
    mutate(forecast = mean(c_across(c(contains("Tuned"))), na.rm=TRUE)) %>% 
    ungroup() %>% 
    mutate(correct = if_else((forecast < 0 & ACTUAL < 0)|
                               (forecast >= 0 & ACTUAL >= 0),1,0)) %>% 
    filter(date == "2024-12-06") %>% View()

#symbol %in% c("NVDA","GE","FI","FIS","CTAS","WELL","MSFT","MMM","IRM","NVDA","TT")

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
  filter(.index == max(.index) & symbol %in% c("NVDA","GE","FI","FIS","CTAS","WELL","CRWD")) %>% 
  select(.model_id, .model_desc, symbol, date, .value, .conf_lo, .conf_hi)

test_fcst %>% filter(!is.na(.model_id)) %>% group_by(symbol) %>% summarise(mean = mean(.value), median = median(.value))
test_fcst %>% filter(is.na(.model_id))

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

model_ensemble_tbl %>% 
    modeltime_calibrate(testing(splits),
                        id = "symbol")

model_ensemble_tbl %>%
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
        actual_data = data_prepared_tbl,
        keep_data = T
    )

# * ensemble forecast test plot ----
forecast_ensemble_test_tbl %>% 
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
data_prepared_clean_tbl <- data_prepared_tbl %>% 
    filter_by_time(date, .start = max(date)-years(2)) %>% 
    filter(symbol %in% forecast_symbols) %>% 
    #droplevels() %>% 
    drop_na()

model_ensemble_refit_tbl <- model_ensemble_tbl_wt %>% 
    modeltime_refit(data_prepared_clean_tbl)

# * Ensemble final forecast ----
final_forecast_tbl <- forecast_tbl %>% filter(symbol %in% forecast_symbols)

model_ensemble_final_forecast <- model_ensemble_refit_tbl %>% 
    modeltime_forecast(
        new_data    = final_forecast_tbl,
        actual_data = data_prepared_clean_tbl,
        keep_data   = T,
        conf_by_id  = T
    )  

# * plot final forecast ----
model_ensemble_final_forecast %>% 
  arrange(desc(date)) %>% 
    group_by(symbol) %>% 
    plot_modeltime_forecast(
        .facet_ncol = 2,
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
