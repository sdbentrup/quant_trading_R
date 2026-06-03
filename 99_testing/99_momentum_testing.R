library(data.table)

setDT(prices_dt)

prices_features_dt[
    order(symbol, date),
    `:=`(
        # Python-style long-term return: (P[t-21] - P[t-252]) / P[t-252]
        ret_252 = shift(close, 21) / shift(close, 252) - 1,
        
        # Python-style short-term return: (P[t] - P[t-21]) / P[t-21]
        ret_21  = shift(close,1) / shift(close, 21) - 1,
        
        # return 1 day
        roc_1 = ROC(close, 1),
        
        # Python-style volatility: sd of simple daily returns over last 126 obs
        vol_126 = frollapply(roc_1, 126, FUN = sd, fill = NA)
    ),
    by = symbol
][
    ,
    momentum := (ret_252 - ret_21) / vol_126
]

setorderv(prices_dt, cols = c("symbol","date"))

prices_dt[, `:=`(
    Close_roc_0_1 = ROC(close, n = 1),
    Close_lag_1   = data.table::shift(close, n = 1),
    Close_lag_21  = data.table::shift(close, n = 21),
    Close_lag_252 = data.table::shift(close, n = 252)
), keyby = symbol]

prices_dt[, Close_momentum_21_252 := ((Close_lag_21 / Close_lag_252) - (Close_lag_1 / Close_lag_21 - 1))]
prices_dt[, Close_momentum_21_252_126 := Close_momentum_21_252/frollapply(Close_roc_0_1, FUN = sd, N = 126)]

prices_features_dt[!is.na(Return_fwd_21),.(date,symbol,Return_fwd_21,Close_momentum_21_252_126,ret_252,ret_21,vol_126,momentum)]
