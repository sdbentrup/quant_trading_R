# R Script for optimizing the portfolio mix for the trading system
# resources ---- 
# https://rossb34.github.io/PortfolioAnalyticsPresentation2017/
# https://github.com/braverock/PortfolioAnalytics
# https://cran.r-project.org/web/packages/PortfolioAnalytics/vignettes/robustCovMatForPA.pdf
# https://rossb34.github.io/PortfolioAnalyticsPresentation 

# libraries ----
library(data.table)
library(PortfolioAnalytics)
library(timetk)
library(tidyquant)
library(DEoptim)
library(CVXR)
library(ROI)
library(doParallel)
library(tidyverse)
library(plotly)
# library(RobStatTM)
# library(GSE)

# import data from saved data ----
model_ensemble_final_forecast <-readRDS("01_save_data/01_saved_forecasts/2026-05-27_model_ensemble_final_forecast.rds")
acc_by_symbol <- readRDS("02_models/2026-05-27_acc_by_symbol.rds")

forecast_acc_symbol <- model_ensemble_final_forecast %>% 
    #filter(date == max(date)) %>% 
    merge(acc_by_symbol)

# select the top n stocks
# select by mean prediction?
stock_picks <- forecast_acc_symbol %>% 
    filter(.key == 'prediction') %>% 
    summarise(mean_pred = mean(.value), .by = symbol) %>% 
    arrange(desc(mean_pred)) %>% 
    merge(acc_by_symbol) %>% 
    #filter(.value >= 0.006) %>% 
    mutate(ev = (1-rmse) * mean_pred) %>% # expected value; not technically an ev but attempts to risk-adjust returns
    slice_max(ev, n = 10) %>% 
    pull(symbol) 

# or by last prediction?
# this seems not to be as reliable as the average and ev method above
# stock_picks <- forecast_acc_symbol %>% 
#   filter(date == max(date) & .value > 0) %>% 
#     #slice_min(rmse, n = 80) %>%
#     slice_max(.value, n = 10) %>%
#     #select(symbol, date, .value, rmse, rsq, ev)
#     pull(symbol)

# get price data for top stocks ----
prices <- tq_get(stock_picks, from = today()-years(4))

# * Review returns by symbol ----
prices %>%
  ggplot(aes(x = date, y = close)) +
  geom_candlestick(aes(open = open, high = high, low = low, close = close),
                   colour_up = "darkgreen", colour_down = "darkred", 
                   fill_up  = "darkgreen", fill_down  = "darkred") +
  labs(y = "Closing Price", x = "") + 
  facet_wrap(~ symbol, scale = "free_y") +
  theme_tq()

# summarize prices to xts ----
returns <- prices %>%
  group_by(symbol) %>%
  tq_transmute(select = close,
               mutate_fun = periodReturn,
               period = 'monthly',
               col_rename = "close_ret") %>% 
  ungroup()


returns_xts <- returns %>%
  drop_na() %>% 
  pivot_wider(id_cols = date, names_from = symbol, values_from = close_ret) %>%
  as.xts()

# * visualize portfolio performance ----
returns %>% 
  tq_portfolio(assets_col   = symbol,
               returns_col  = close_ret,
               #weights      = wts,
               col_rename   = "investment.growth",
               wealth.index = T) %>%
  mutate(investment.growth = investment.growth * 10000) %>% 
  plot_ly(x = ~date, y = ~investment.growth, type = "scatter", mode = "lines") 

returns %>% 
  group_by(symbol) %>%
  tq_performance(Ra = close_ret,
                 performance_fun = SharpeRatio,
                 Rf = 0.04/12)

returns %>% 
    tq_performance(Ra = close_ret,
                   performance_fun = SharpeRatio,
                   # performance_fun = CalmarRatio,
                   Rf = 0.04/12)

# optimize current portfolio risk parity ----

# * setup optimization ----
# global minimum variance long only portfolio
port_spec <- portfolio.spec(assets = colnames(returns_xts))
port_spec <- add.constraint(port_spec, type = "weight_sum", min_sum=0.99, max_sum=1.01)
port_spec <- add.constraint(port_spec, type = "box", min = 0.04, max = 0.25)
# port_spec <- add.constraint(portfolio = port_spec, type = "long_only") # only positive weights
# port_spec <- add.constraint(portfolio = port_spec, type="transaction_cost", ptc=0.05/100)
# port_spec <- add.objective(portfolio = port_spec, type = "risk", name = "StdDev")
# port_spec <- add.constraint(port_spec, type = "full_investment") 

port_spec <- add.objective(port_spec, type = "return", name = 'mean')
port_spec <- add.objective(port_spec, type = "risk", name = "ES")
port_spec <- add.objective(port_spec, type = "risk_budget", name = "ES", min_concentration=TRUE) # for a min-var portfolio use momentsFUN custom.covRob.x


# * optimize portfolio ----

registerDoParallel(cores = 6)
port_opt_rebal <- optimize.portfolio.rebalancing(returns_xts,
                                                 portfolio = port_spec,
                                                 # optimize_method = "DEoptim",trace = T,traceDE=10,
                                                 # search_size = 20000,
                                                 optimize_method = "CVXR",
                                                 # momentFUN = "custom.covRob.Mcd",#"custom.covRob.TSGS","set.portfolio.moments", #
                                                 # arguments = list("boudt"), #meucci black_litterman
                                                 rebalance_on = "months",
                                                 # training_period = 100 #since weekly data
                                                 #moving_window = 100 #since weekly data
                                                 rolling_window = 6 # N of monthly data
                                                 )

port_opt <- optimize.portfolio(returns_xts,
                               portfolio = port_spec,
                               #rp = sp500_rp,
                               # optimize_method = "CVXR",
                               optimize_method = "ROI",
                               # optimize_method = "DEoptim", traceDE=10,
                               # search_size = 20000,
                               #, "random", "ROI", "ROI_old", "pso", "GenSA","CVXR"
                               # momentFUN = "custom.covRob.Mcd",#"custom.covRob.TSGS","custom.covRob.MM"
                               # arguments = list("black_litterman"),
                               # arguments = list("boudt"),
                               trace = TRUE
                               )


stopImplicitCluster()
port_opt
port_opt_rebal

port_opt[2:4]

opt.outputMvo(port_opt, returns_xts, digits = 3, frequency="monthly")

# view backtest results
# Extract time series of portfolio weights 
wts.rebal <- extractWeights(port_opt_rebal) 
wts.rebal <- wts.rebal[complete.cases(wts.rebal),]

# Compute cumulative returns of portfolio 
CSM <- Return.rebalancing(returns_xts, wts.rebal) 
backtest.plot(wts.rebal)
backtest.plot(wts.rebal, plotType = 'cumRet')

names(port_opt_rebal)
names(port_opt_rebal$opt_rebalancing)

summary(port_opt)
summary(port_opt_rebal)
extractObjectiveMeasures(port_opt)
extractObjectiveMeasures(port_opt_rebal)
chart.RiskBudget(port_opt)
#plot(port_opt, risk.col = "CVaR", neighbors = 3)

chart.RiskReward(port_opt,
                 risk.col = "ES",
                 return.col = "mean",
                 chart.assets = T)

barplot(extractWeights(port_opt), 
        main ="Optimal Portfolio Asset Allocation", 
        col  =rainbow(length(extractWeights(port_opt)), alpha = 0.7))

# set optimal weights ----
opt_weights <- extractWeights(port_opt) %>%
  enframe() %>%
  rename("weight"="value") 

opt_weights %>%
  mutate(value = weight*30000) %>%
  arrange(name) %>% 
  print(n = nrow(.))

extractWeights(port_opt) %>%
  enframe() %>%
  rename("weight"="value") %>%
  plot_ly(x = ~fct_reorder(name,weight,.desc = T),
          y = ~weight, color = ~name, type = "bar", alpha = 0.8,
          text = ~scales::percent(weight,0.1), textposition = "outside") %>% 
  layout(showlegend = F, xaxis = list(title = "symbol"))

port_weights_tbl <- extractWeights(port_opt_rebal) %>%
  fortify.zoo() %>%
  pivot_longer(-Index,names_to = "symbol", values_to = "weight") %>%
  rename("date"="Index")

port_weights_tbl %>% 
  plot_time_series(.date_var = date,
                   .value = weight,
                   .color_var = symbol,
                   .smooth = F,
                   .line_alpha = 0.7)

port_weights_tbl %>%
  filter(date == max(date)) %>%
  mutate(value = weight*31000) %>%
  arrange(desc(value))

chart.Weights(port_opt)

chart.RiskReward(port_opt,
                 risk.col = "ES",
                 return.col = "mean",
                 chart.assets = T)

# unweighted portfolio returns
port_base_return <- Return.portfolio(returns_xts)
table.AnnualizedReturns(port_base_return)

# adjusted portfolio returns with optimal weights
port_returns <- Return.portfolio(returns_xts,
                                 weights = extractWeights(port_opt))
port_returns_rebal <- Return.portfolio(returns_xts,
                                       weights = extractWeights(port_opt_rebal))

table.AnnualizedReturns(port_returns)
charts.PerformanceSummary(port_returns)

ports <- cbind(port_base_return, port_returns, port_returns_rebal) #,port_port_returns_ma,
colnames(ports) <- c("base","optimized","rebalancing") #"ma_test",
table.AnnualizedReturns(ports, Rf = 0.05/12)
charts.PerformanceSummary(ports, Rf = 0.05/12)

# save portfolio optimization ----
write_rds(port_opt,str_glue("01_save_data/02_portfolios/{today()}_port_opt.rds"))
