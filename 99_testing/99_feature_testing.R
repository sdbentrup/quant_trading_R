library(data.table)
library(tidymodels)
library(tidyverse)
library(timetk)
library(future)
library(future.apply)
library(ranger)
library(plotly)

# MACD length testing ----
# select some random symbols
dt <- prices_features_dt[symbol %in% c("ABBV","ZTS","WMB","NKE","EOG","ITW","AAPL","MMM") & 
                             !is.na(Return_fwd_21),
                         .(symbol,date, close, Return_fwd_21, SAR, Close_macd_short, Close_macd_short_signal)] %>% droplevels()

setorderv(dt, c("symbol","date"))

# short macd_test
# parameter vectors
is <- seq(3, 21, 3)
js <- c(6,12, 19, 21, 26, 30, 40, 52) # 52 showed better results
ks <- seq(3, 21, 3)

# grid of combinations
combos <- CJ(i = is, j = js, k = ks)

results <- rbindlist(
    future_lapply(seq_len(nrow(combos)), function(idx) {
        i <- combos$i[idx]; j <- combos$j[idx]; k <- combos$k[idx]
        tmp <- dt[date >= "2025-01-01"]                      # avoid mutating original
        tmp[, close_macd := TTR::EMA(close, n = i) - TTR::EMA(close, n = j), by = symbol]
        tmp[, close_macd_signal := TTR::EMA(close_macd, n = k), by = symbol]
        # run cor_test on the table without the 'close' column (as in your example)
        res <- cor_test(tmp[, !'close'], Return_fwd_21)
        vals <- res %>% select(var2, cor) %>% pivot_wider(var2,cor)
        data.table(i = i, j = j, k = k,vals)
    })
)

results %>% arrange(close_macd_signal,close_macd) #%>% View()

# test parameters 1
dt[, ":=" (
    close_macd_test = EMA(close, 19) - EMA(close, 39)),
   by = symbol][,":=" (close_macd_signal_test = EMA(close_macd_test, 18)),
                by = symbol]

dt[,!c("close")] %>% cor_test(Return_fwd_21)

dt[date >= "2025-12-01"] %>% plot_ly(x = ~close_macd_test, 
                                     y = ~Return_fwd_21, 
                                     color = ~symbol,
                                     type = "scatter")

lm(Return_fwd_21 ~ close_macd_signal_test + close_macd_test + SAR, dt %>% drop_na()) %>% summary()
lm(Return_fwd_21 ~ Close_macd_short + Close_macd_short_signal + SAR, dt) %>% summary()

library(ranger)
ranger_test <- ranger(Return_fwd_21 ~ close_macd_signal_test + close_macd_test + SAR, 
       dt %>% drop_na(), 
       importance = "permutation") 

ranger_test %>% importance() %>% enframe() %>% arrange(desc(value))

ranger <- ranger(Return_fwd_21 ~ Close_macd_short + Close_macd_short_signal + SAR, 
       dt %>% drop_na(), 
       importance = "permutation") 

ranger %>% importance() %>% enframe() %>% arrange(desc(value))

ranger(Return_fwd_21 ~ . -symbol -close -date, 
    dt %>% drop_na(), importance = "impurity_corrected") %>% importance() %>% enframe() %>% arrange(desc(value))

library(rpart)
library(rpart.plot)

rpart(Return_fwd_21 ~ . -symbol -close -date, 
             dt %>% drop_na()) %>% rpart.plot(fallen.leaves = F)

# test parameters 2
dt[, ":=" (
    close_macd_base = EMA(close, 12) - EMA(close, 26))
][,":=" (close_macd_base_signal = EMA(close_macd, 9)), by = symbol]

dt[,!c("close")] %>% cor_test(Return_fwd_21)

dt[date >= "2025-12-01"] %>% plot_ly(x = ~close_macd_base_signal, 
                                     y = ~Return_fwd_21, color = ~symbol,
                                     type = "scatter")

lm(Return_fwd_21 ~ close_macd_base_signal + close_macd_base, dt[date >= "2025-07-01",]) %>% summary()

ranger(Return_fwd_21 ~ close_macd_base_signal + close_macd_base, 
       dt[date >= "2025-07-01",], 
       importance = "permutation") %>% importance() %>% enframe() %>% arrange(desc(value))

# rf testing
is <- c(9, 12, 18)
js <- c(21, 26, 30)
ks <- c(9, 12, 18)

# grid of combinations
combos <- CJ(i = is, j = js, k = ks)

results_rf <- rbindlist(
    future_lapply(seq_len(nrow(combos)), function(idx) {
        i <- combos$i[idx]; j <- combos$j[idx]; k <- combos$k[idx]
        tmp <- dt[date >= "2025-01-01",.(symbol,date, close, Return_fwd_21)]  # avoid mutating original
        tmp[, close_macd := TTR::EMA(close, n = i) - TTR::EMA(close, n = j), by = symbol]
        tmp[, close_macd_signal := TTR::EMA(close_macd, n = k), by = symbol]
        res <- ranger(Return_fwd_21 ~ close_macd+close_macd_signal, tmp)
        rsq <- res$r.squared
        err <- res$prediction.error
        data.table(i = i, j = j, k = k, rsq = rsq, rmse = err)
    }, future.seed = 101)
)
saveRDS(results_rf,"01_save_data/results_rf.rds")

results_rf %>% View()

# long testing
# example parameter vectors
is <- c(seq(50, 100, 50))
js <- c(200,300) # 52 showed better results
ks <- seq(30,90,30)

# grid of combinations
combos <- CJ(i = is, j = js, k = ks)

results <- rbindlist(
    lapply(seq_len(nrow(combos)), function(idx) {
        i <- combos$i[idx]; j <- combos$j[idx]; k <- combos$k[idx]
        tmp <- copy(dt)                      # avoid mutating original
        tmp[, close_macd := TTR::EMA(close, n = i) - TTR::EMA(close, n = j), by = symbol]
        tmp[, close_macd_signal := TTR::EMA(close_macd, n = k), by = symbol]
        # run cor_test on the table without the 'close' column (as in your example)
        res <- cor_test(tmp[, !'close'], Return_fwd_21)
        vals <- res %>% select(var2, cor) %>% pivot_wider(var2,cor)
        data.table(i = i, j = j, k = k,vals)
    })
)

results %>% arrange(close_macd_signal,close_macd) #%>% View()

# test parameters 1
dt[, ":=" (
    close_macd = EMA(close, 50) - EMA(close, 300))
][,":=" (close_macd_signal = EMA(close_macd, 30))]

dt[,!c("close")] %>% cor_test(Return_fwd_21)

dt[date >= "2025-12-01"] %>% plot_ly(x = ~close_macd_signal, 
                                     y = ~Return_fwd_21, 
                                     type = "scatter")

lm(Return_fwd_21 ~ ., dt[date >= "2025-12-01",!c("close","date")]) %>% summary()
ranger(Return_fwd_21 ~ ., dt[date >= "2025-12-01",!c("close","date")])


# RSI testing ----
dt_base <- prices_features_dt[symbol %in% c("ABBV","ZTS","WMB","NKE","EOG","ITW","AAPL") & 
                             !is.na(Return_fwd_21),
                         .(symbol,date, close, Return_fwd_21)] %>% droplevels()
dt <- copy(dt_base)
dt[, rsi := RSI(close, n = 70), keyby = symbol]
dt[, rsi_shift := shift(rsi, 18), keyby = symbol]
dt[, rsi_roc := ROC(rs, 18), keyby = symbol]

# dt %>% cor_test(rsi_21, Return_fwd_21)
dt %>% cor_test(Return_fwd_21)
ranger(Return_fwd_21 ~ . -close -symbol, dt)

dt %>% 
    group_by(symbol) %>% 
    plot_acf_diagnostics(date, Return_fwd_21, .ccf_vars = rsi_21, .show_ccf_vars_only = T)

dt %>% plot_ly(x = ~rsi, y = ~Return_fwd_21, color = ~symbol)

# example parameter vectors

#js <- c(200,300) # 52 showed better results
#ks <- seq(30,90,30)

# grid of combinations
combos <- CJ(i = is, j = js, k = ks)

rsi_grid <- expand.grid(n <- seq(7,70,7)) %>% as.data.table()

for (n in 1:nrow(rsi_grid)) {
        n <- n
        tmp <- copy(dt_base) # avoid mutating original
        tmp[, rsi := RSI(close, n = n), keyby = symbol]

        res <- tmp %>% drop_na() %>% cor_test(rsi, Return_fwd_21)
        vals <- res %>% select(cor)
        rsi_grid[n, cor := vals]
    }
rsi_grid

