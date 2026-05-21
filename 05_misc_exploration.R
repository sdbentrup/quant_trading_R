get <- forecast_acc_sybmol %>% 
    select(symbol, date, .value, rmse, rsq, ev) %>% 
    # filter(rmse < 0.05) %>% 
    # filter(.value > 0) %>% 
    # slice_min(rmse, n = 50) %>% 
    # slice_max(.value, n = 10) %>% 
    # mutate(ev = (1-rmse) * .value) %>% 
    slice_max(ev, n = 10) %>% 
    pull(symbol)

train_date <- today() - years(2)-months(6)

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
