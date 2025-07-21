# Make orders ----
# For an order we need to know
# how many shares to buy
# For this we need the budget (i.e. percent of portfolio)
# and the price of the shares
# and the existing size if any of the position
# Share we get from the allocation in the optimization
# for price we need to get the current price
# the order is then (portfolio_val * allocation/pricePershare)-existing shares
# set this as the target shares
# transaction shares == target_shares - held_shares
# compare number of shares to transact to number of shares held if any
# if target == 0 then action == "sell" and qty == transaction_shares
# if shares_held > shares_order then action == "sell" and qty == transaction_shares

# Install packages ----
#install.packages("IBrokers")
library(IBrokers)
library(tidyverse)
library(plotly)
library(data.table)
library(PortfolioAnalytics)
library(timetk)
library(tidyquant)
library(doParallel)

# Import data from forecast ----
#change to the most recent forecast saved
model_ensemble_final_forecast <- read_rds("04-Financial/04_01_save_data/2025-07-18_model_ensemble_final_forecast.rds")

model_ensemble_final_forecast %>% 
  arrange(desc(date)) %>% 
  dplyr::select(symbol, .value, date)

model_ensemble_final_forecast %>% 
  filter(date == max(date)) %>% 
  slice_max(.value, n = 10) %>% 
  select(symbol, .value)

# select the top n stocks
stock_picks <- model_ensemble_final_forecast %>% 
  filter(date == max(date)) %>% 
  slice_max(.value, n = 10) %>% 
  pull(symbol) %>% 
  as.character()

# IBrokers connection ----
tws = twsConnect(port = 7497, clientId = 12) #paper trading port 7497; live 7496?
isConnected(tws)

# Portfolio query ----
a <- reqAccountUpdates(tws,subscribe = T)

value <- twsPortfolioValue(a, zero.pos=TRUE)

port <- as.data.table(value)
port

setnames(port, "local","symbol")

port[,":=" (
  current_weight = marketValue/sum(marketValue),
  starting_weight = averageCost/sum(averageCost)
)]

port

port_value <- as.numeric(a[[1]][["CashBalance"]][["value"]])+sum(port$marketValue)#as.numeric(a[[1]][["GrossPositionValue"]][["value"]])

# compare holdings to predicted values
merge(port[,.(symbol, unrealizedPNL)],
      model_ensemble_final_forecast %>% 
        filter(date == max(date)) %>% 
        select(symbol, .value))

# get pricing for list of stock_picks ----

# Define function to get market data for a vector of stock symbols
get_market_prices <- function(symbols, tws) {
  prices <- list()  # Initialize an empty list
  
  for (symbol in symbols) {
    # Request market data
    data <- reqHistoricalData(tws, twsSTK(symbol), barSize = "1 min", duration = "1 D")
    
    # Convert to data.table
    data.dt <- as.data.table(data)
    
    # Ensure the index column is used properly
    lastPrice <- as.numeric(data.dt[which.max(data.dt[, index]), ][, 5])
    
    # Store result in list
    prices[[symbol]] <- list(symbol = symbol, lastPrice = lastPrice)
  }
  
  return(rbindlist(prices, use.names = TRUE, fill = TRUE))  # Convert list to data.table
}

# Apply function across stock symbols

stock_prices_list <- future_lapply(stock_picks, get_market_prices, tws = tws)

# Combine results into a single data.table
final_prices <- rbindlist(stock_prices_list, use.names = TRUE, fill = TRUE)

final_prices

# create table of stocks to buy

stocks_table <- copy(final_prices)

# CREATE function for equal weighting or weighting from optimization?

# equal weight
stocks_table[,target_shares := floor((port_value/length(stock_picks))/lastPrice)]
stocks_table[,.(value = lastPrice*target_shares)]

# optimiztion weights
stocks_table <- merge(stocks_table,
                      opt_weights,
                      by.x = "symbol",
                      by.y = "name") %>% 
  setDT()

stocks_table[,target_shares := floor((port_value*weight)/lastPrice)]

# create table of portfolio
port_table <- data.table(port[,.(symbol,position, marketValue, unrealizedPNL)])

# merge tables
actions_table <- merge(port_table, stocks_table, all = T)

# fill na with 0
setnafill(actions_table, "const", 0, 
          cols = names(actions_table[,which(sapply(.SD, is.numeric))]))

# create actions
# sell if the target is 0
# if we are decreasing positions then also sell
# otherwise BUY, BUY, BUY!

actions_table[,action := fcase(target_shares == 0, "SELL",
                               target_shares < position, "SELL",
                               default = "BUY")]

# set the shares target based on increasing the position for buying otherwise decreasing if selling
actions_table[,qty := fifelse(action == "BUY",
                              target_shares-position,
                              position-target_shares)]

# calculate the new value, for reference only
actions_table[,new_mkt_value := lastPrice*target_shares]

# set the order to reverse so that sales happen before buys
setorder(actions_table, -action)

actions_table
actions_table[,.(sum(new_mkt_value, na.rm = T))]
actions_table[action == "SELL",.(symbol,unrealizedPNL*(qty/position))]
actions_table[action == "SELL",.(sum(unrealizedPNL*(qty/position)))]

actions_table[action == "SELL",.(sum(unrealizedPNL*(qty/position)))]/sum(actions_table$marketValue)

port_value-actions_table[,.(sum(new_mkt_value, na.rm = T))]

# orders ----
# how to do with multiple contracts? for loop or lapply?
# placeOrder(twsconn=tws, Contract=twsSTK("AAPL"), Order=twsOrder(reqIds(tws), "BUY", 10, "MKT"))

# Define the order processing function
order_function <- function(actions_table, tws) {
  # Ensure the table is a data.table
  setDT(actions_table)
  
  # Iterate through each row in the table
  for (i in 1:nrow(actions_table)) {
    symbol <- actions_table[i, symbol]
    action <- actions_table[i, action]
    qty <- actions_table[i, qty]
    
    # Place the order
    placeOrder(
      twsconn = tws,
      Contract = twsSTK(symbol),
      Order = twsOrder(reqIds(tws), action, qty, "MKT")
    )
  }
}

# * submit orders ----
order_function(actions_table, tws)

# write order history ----
write_rds(actions_table, str_glue("04-Financial/04_01_save_data/{today()}_actions_table.rds"))

# close the tws session ----
twsDisconnect(tws)
