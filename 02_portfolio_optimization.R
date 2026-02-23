# R Script for optimizing the portfolio mix for the trading system
# resources:
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
# library(RobStatTM)
# library(GSE)

# import data from saved data ----
model_ensemble_final_forecast <- read_rds("04-Financial/04_01_save_data/2025-07-11_model_ensemble_final_forecast.rds")

model_ensemble_final_forecast %>%
  filter(date == max(date) & .value > 0) %>%
  #slice_max(.value, n = 15) %>% 
  dplyr::select(symbol, .value) %>% 
  arrange(desc(.value))

# select the top n stocks
stock_picks <- model_ensemble_final_forecast %>% 
  filter(date == max(date) & .value > 0) %>% 
  slice_max(.value, n = 10) %>% 
  pull(symbol) %>% 
  as.character()

# get price data for top stocks ----
prices <- tq_get(stock_picks, from = today()-years(6))

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
               period = 'weekly',
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
               wealth.index = TRUE) %>%
  mutate(investment.growth = investment.growth * 10000) %>% 
  plot_ly(x = ~date, y = ~investment.growth, type = "scatter", mode = "lines") 

returns %>% 
  group_by(symbol) %>%
  tq_performance(Ra = close_ret,
                 performance_fun = SharpeRatio,
                 Rf = 0.05/12)
# optimize current portfolio risk parity ----

# * setup optimization ----
# global minimum variance long only portfolio
port_spec <- portfolio.spec(assets = colnames(returns_xts))
port_spec <- add.constraint(port_spec, type = "weight_sum", min_sum=0.99, max_sum=1.01)
port_spec <- add.constraint(port_spec, type="box", min = 0.05, max = 0.35)
# port_spec <- add.constraint(portfolio = port_spec, type = "long_only") # only positive weights
# port_spec <- add.constraint(portfolio= port_spec, type="transaction_cost", ptc=0.05/100)
#port_spec <- add.objective(portfolio = port_spec, type = "risk", name = "StdDev")
# port_spec <- add.constraint(port_spec, type="full_investment") 

port_spec <- add.objective(port_spec, type = "return", name = 'mean')
port_spec <- add.objective(port_spec, type="risk", name="ES")
port_spec <- add.objective(port_spec, type="risk_budget", name="ES", min_concentration=TRUE) # for a min-var portfolio use momentsFUN custom.covRob.x


# * optimize portfolio ----

registerDoParallel(cores = 4)
port_opt_rebal <- optimize.portfolio.rebalancing(returns_xts,
                                                 portfolio = port_spec,
                                                 optimize_method = "DEoptim",
                                                 # search_size = 20000,
                                                 trace = T,traceDE=10,
                                                 # momentFUN = "custom.covRob.Mcd",#"custom.covRob.TSGS","set.portfolio.moments", #
                                                 #arguments = list("boudt"), #meucci black_litterman
                                                 rebalance_on = "months",
                                                 # training_period = 100 #since weekly data
                                                 #moving_window = 100 #since weekly data
                                                 rolling_window = 100 # 2 years of weekly data
                                                 )

port_opt <- optimize.portfolio(returns_xts,
                               portfolio = port_spec,
                               #rp = sp500_rp,
                               optimize_method = "DEoptim", 
                               # search_size = 20000,
                               #, "random", "ROI", "ROI_old", "pso", "GenSA","CVXR"
                               # momentFUN = "custom.covRob.Mcd",#"custom.covRob.TSGS","custom.covRob.MM"
                               #arguments = list("black_litterman"),
                               trace=TRUE, traceDE=10)


stopImplicitCluster()
port_opt
port_opt_rebal

port_opt[2:4]

opt.outputMvo(port_opt, returns_xts, digits = 3, frequency="weekly")

# view backtest results
# Extract time series of portfolio weights 
wts.rebal <- extractWeights(port_opt_rebal) 
wts.rebal <- wts.rebal[complete.cases(wts.rebal),]

# Compute cumulative returns of portfolio 
CSM <- Return.rebalancing(returns_xts, wts.rebal) 
backtest.plot(wts.rebal)


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
        main="Optimal Portfolio Asset Allocation", 
        col=rainbow(length(extractWeights(port_opt))))

# set optimal weights ----
opt_weights <- extractWeights(port_opt) %>%
  enframe() %>%
  rename("weight"="value") 

opt_weights %>%
  mutate(value = weight*100000) %>%
  arrange(name) %>% 
  print(n = nrow(.))

extractWeights(port_opt) %>%
  enframe() %>%
  rename("weight"="value") %>%
  plot_ly(x = ~fct_reorder(name,weight,.desc = T),
          y = ~weight, color = ~name, type = "bar", alpha = 0.8) %>% 
  layout(showlegend = F, xaxis = list(title = "symbol"))

port_weights_tbl <- extractWeights(port_opt_rebal) %>%
  fortify.zoo() %>%
  pivot_longer(-Index,names_to = "symbol", values_to = "weight") %>%
  rename("date"="Index")

port_weights_tbl %>% 
  plot_time_series(.date_var = date,
                   .value = weight,
                   .color_var = symbol,
                   .smooth = F)

port_weights_tbl %>%
  filter(date == max(date)) %>%
  mutate(value = weight*port_value) %>%
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
write_rds(port_opt,str_glue("04-Financial/04_01_save_data/{today()}_port_opt.rds"))
