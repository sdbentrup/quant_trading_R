# Backtesting ----
# libraries ----
library(data.table)
library(PortfolioAnalytics)
library(timetk)
library(tidyquant)
library(doParallel)
# library(RobStatTM)
# library(GSE)

# import data from saved data ----
full_data <- read_rds("01_save_data/2025-11-05_full_data.rds")

model_ensemble_final_forecast <- read_rds("04-Financial/04_01_save_data/2025-07-11_model_ensemble_final_forecast.rds")

# market data for a baseline ----
sp <- tq_get(x = "^GSPC", get = "stock.prices", from = today() - years(6),complete_cases = T)
sp_returns <- sp %>%
    tq_transmute(select = adjusted,
                 mutate_fun = periodReturn,
                 period = 'monthly',
                 col_rename = "return_month")

sp_xts <- as.xts(sp_returns)

# create backtesting data ----
backtest_full_data <- full_data[date %in% full_data[, .(date = max(date)), by = .(year = year(date), month = month(date))][,date]]
backtest_full_data <- backtest_full_data[symbol %in% unique(model_ensemble_final_forecast$symbol)] %>% drop_na() %>% droplevels()

backtest_full_data %>% summary()

forecast_ensemble_backtest <- model_ensemble_refit_tbl %>% #modeltime_table(wflw_fit_lgb_tuned) %>% #model_ensemble_tbl_wt %>% 
    modeltime_forecast(
        new_data    = backtest_full_data,
        #actual_data = data_prepared_tbl,
        keep_data   = T,
        conf_by_id  = T
    )

setDT(forecast_ensemble_backtest)

# add the signals  to the prices data
prices_signal <- merge(prices_base[date %between% c(min(forecast_ensemble_backtest$date),
                                                   max(forecast_ensemble_backtest$date)+1),
                                   .(symbol, date, close)],
                       forecast_ensemble_backtest[,.(symbol, date, .value)],
                       all.x = F) 

# create the trading signal based on positive expected returns and within the top n of predictions
# change the rank here to create portfolio with more or less diversification
prices_signal[,rank := rank(-.value), 
              keyby = date][, signal := fifelse(.value > 0 & between(rank,1,10), 1, 0)]#[, signal := fifelse(.value > 0 & rank <= 20, 1, 0)]


# # fill signals forward
# prices_signal[,signal := nafill(signal, type = "locf"), keyby = symbol]

# create weights by month
prices_signal[, weight := signal/sum(signal), keyby = date]

# lag weight by 1 period for trading month
# this is the correct weight since it shows the weight in the month after the prediction was made
prices_signal[,weight_lag := data.table::shift(weight, 1), keyby = symbol]

# calculate returns
prices_signal[, return := close/(data.table::shift(close, 1))-1, keyby = symbol]#[,weighted_return := weight_lag * return]

setnafill(prices_signal, type = "const", fill = 0, cols = "return")

# prices_signal[,month_return := (return*weight*100000)]

# prices_signal[,.(total = sum(month_return)), keyby = date]

# calculate returns
returns.xts <- prices_signal[,dcast(.SD, date ~ symbol, value.var = "return", fill = 0)] %>% as.xts()
weights.xts <- prices_signal[,dcast(.SD, date ~ symbol, value.var = "weight_lag", fill = 0, subset = .(!is.na(weight_lag)))] %>% as.xts()

port.return <- Return.portfolio(returns.xts,
                 weights = weights.xts,
                 geometric = F,
                 value = 30000)

port.return_wi <- Return.portfolio(returns.xts,
                                weights = weights.xts,
                                geometric = F,
                                rebalance_on = "months",
                                wealth.index = T)

port.return_wi %>% fortify.zoo() %>% mutate(value = portfolio.wealthindex*30000)

# create a portfolio of the same assets without any weighting for comparison how they would do to just buy those assets
base.return <- Return.portfolio(returns.xts,value = 30000, rebalance_on = "months")

table.AnnualizedReturns(port.return)
charts.PerformanceSummary(port.return)
backtest.plot(port.return)

port.return %>%
  fortify.zoo() %>% 
  plot_ly(x = ~Index, y  = ~portfolio.returns, type = "scatter", mode = "lines")

port.return_wi %>%
  fortify.zoo() %>% 
  plot_ly(x = ~Index, y  = ~portfolio.wealthindex, type = "scatter", mode = "lines")

VaR(port.return)
VaR(base.return)

UpsideRisk(port.return)
UpsideRisk(base.return)

test <- cbind(sp_xts,base.return, port.return)
names(test) <- c("market","base","portfolio")
table.AnnualizedReturns(test)
charts.PerformanceSummary(test)

# view returns by symbol
prices_signal[weight != 0] %>%
  droplevels() %>% 
  plot_ly(x = ~date, y = ~weight, color = ~symbol, type = "bar") %>% layout(barmode = "stack")

prices_signal[,.(weight_sum = sum(weight)), symbol] %>% arrange(desc(weight_sum))
prices_signal[,.(weight_sum = sum(weight)), date] %>% arrange(desc(weight_sum))

# calculate some metrics on the predictions
prices_signal[,error := .value - shift(return,-1)]

prices_signal %>% mutate(return = shift(return,-1)) %>% cor_test(return, .value)
prices_signal[signal == 1,] %>% mutate(return = shift(return,-1)) %>% cor_test(return, .value)

prices_signal[,.(rmse = sqrt(mean(error^2, na.rm = T)))]
prices_signal[signal == 1,.(rmse = sqrt(mean(error^2)))]
prices_signal[signal == 1,.(mae = mean(error))]

prices_signal[signal == 1,] %>% plot_ly(x = ~error, type = "histogram")

prices_signal[signal == 1 & error > 0.2,]
prices_signal[signal == 1 & error > 0.2,.N,keyby = rank]

rsq(prices_signal %>% mutate(return = shift(return,-1)), truth = return, estimate = .value)
rsq(prices_signal[signal == 1,], truth = return, estimate = .value)

# try the quantstrat backtesting ----

library(blotter)
library(foreach)
library(quantstrat)

startDate = from
initEq = 1000000
portfolio.st = 'stratML'
account.st = 'acctML'

# Remove the existing strategy if it exists
rm.strat(stratML)

# Initialize the portfolio
initPortf(stratML, symbols = prices_signal$symbol, initDate = initdate, currency = "USD")

# Initialize the account
initAcct(acctML, portfolios = stratML, initDate = initdate, currency = "USD", initEq = initeq)

# Initialize the orders
initOrders(portfolio.st, initDate = initdate)

# Store the strategy
strategy(stratML, store = TRUE)

stratML <- add.signal(strategy = stratML,name="sigCrossover",arguments = list(columns=c("indicator"), relationship="gte"),label="signal")
#stratML <- add.signal(strategy = stratML,name="sigCrossover",arguments = list(column=c("ma50","ma200"),relationship="lt"),label="ma50.lt.ma200")

stratML <- add.rule(strategy = stratML,name='ruleSignal', arguments = list(sigcol="signal",sigval="1", orderqty=100, ordertype='market', orderside='long'),type='enter')
stratML <- add.rule(strategy = stratML,name='ruleSignal', arguments = list(sigcol="signal",sigval="0", orderqty='all', ordertype='market', orderside='long'),type='exit')

out<-applyStrategy(strategy=stratML , portfolios=portfolio.ml)

# try backtesting with tidyquant ----
prices_signal

prices_signal %>% 
  mutate(signal_test = )
