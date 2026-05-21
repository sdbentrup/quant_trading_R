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

# Load packages ----
#install.packages("IBrokers")
library(IBrokers)
library(tidyverse)
library(future.apply)
library(plotly)
library(data.table)
library(PortfolioAnalytics)
library(timetk)
library(tidyquant)
library(doParallel)

# Import data from forecast ----
#change to the most recent forecast saved
model_ensemble_final_forecast <- read_rds("01_save_data/01_saved_forecasts/2026-05-27_model_ensemble_final_forecast.rds")
acc_by_symbol <- read_rds("02_models/2026-05-27_acc_by_symbol.rds")

forecast_acc_sybmol <- model_ensemble_final_forecast %>% 
    filter(date == max(date)) %>% 
    merge(acc_by_symbol) %>% 
    #filter(.value >= 0.006) %>% 
    mutate(ev = (1-rmse) * .value) 

forecast_acc_sybmol %>% 
    slice_max(.value, n = 12) %>% 
    select(symbol, .value, rmse, mae, rsq, ev)

forecast_acc_sybmol %>% 
    filter(.value >0) %>% 
    slice_min(rmse, n = 12) %>% 
    #slice_max(.value, n = 10) %>%
    select(symbol, .value, rmse, mae, rsq, ev)

# select the top n stocks
stock_picks <- forecast_acc_sybmol %>% 
    select(symbol, date, .value, rmse, rsq, ev) %>% 
    # filter(rmse < 0.05) %>% 
    # filter(.value > 0) %>% 
    slice_min(rmse, n = 80) %>%
    slice_max(.value, n = 10) %>%
    # mutate(ev = (1-rmse) * .value) %>% 
    # slice_max(ev, n = 10) %>% 
    pull(symbol) %>% 
    as.character()

# IBrokers connection ----
tws = twsConnect(port = 7496, clientId = 11) #paper trading port 7497; live 7496
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

holdings <- port$symbol

# save portfolio snapshot
write_rds(port, str_glue("01_save_data/02_portfolios/{today()}_live_port.rds"))

port_value <- as.numeric(a[[1]][["NetLiquidation"]][["value"]])#as.numeric(a[[1]][["CashBalance"]][["value"]])+sum(port$marketValue)#as.numeric(a[[1]][["GrossPositionValue"]][["value"]])

# compare holdings to predicted values to see what current holdings are expected to do
merge(port[,.(symbol, unrealizedPNL)],
      model_ensemble_final_forecast %>% 
        filter(date == max(date)) %>% 
        select(symbol, .value))

# get pricing for list of stock_picks ----

# Define function to get market data for a vector of stock symbols
get_market_prices <- function(symbols, tws) {
  prices <- list()  # Initialize an empty list
  
  for (symbol in symbols) {
      
      symbol <- gsub(symbol, pattern = "[-]", replacement = " ")
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

# ** OPTION 1  equal weight
stocks_table[,target_shares := floor((port_value/length(stock_picks))/lastPrice)]
stocks_table[,.(value = lastPrice*target_shares)]

# ** OPTION 2 optimiztion weights
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

actions_table[,":=" (
    limit_price = fifelse(action == "BUY", round(lastPrice * 1.15,2),0),
    stop_price  = fifelse(action == "BUY", round(lastPrice * 0.97,2),0)
)]

# calculate the new value, for reference only
actions_table[,new_mkt_value := lastPrice*target_shares]

# set the order to reverse so that sales happen before buys
setorder(actions_table, -action)

actions_table
actions_table[,.(prev_mkt = sum(marketValue, na.rm = T), new_mkt = sum(new_mkt_value, na.rm = T))]
actions_table[action == "SELL",.(symbol,unrealizedPNL*(qty/position))]
actions_table[action == "SELL",.(sum(unrealizedPNL*(qty/position)))]

actions_table[action == "SELL",.(sum(unrealizedPNL*(qty/position)))]/sum(actions_table$marketValue)

port_value-actions_table[,.(sum(new_mkt_value, na.rm = T))]

# orders ----
# how to do with multiple contracts? for loop or lapply?
# placeOrder(twsconn=tws, Contract=twsSTK("AAPL"), Order=twsOrder(reqIds(tws), "BUY", 10, "MKT"))

# test ordering function

# symbol <- "AAPL"
# action <- "BUY"
# qty    <- 2
# 
# tp_price  <- round(309.19*1.15,2)
# sl_price  <- round(309.19*0.97,2)
# 
# # Parent order (market buy)
# parent_id <- as.numeric(reqIds(tws))
# parent <- twsOrder(
#     orderId   = parent_id,
#     action    = "BUY",
#     totalQuantity = qty,
#     orderType = "MKT",
#     transmit  = FALSE
# )
# # Take‑profit limit order
# tp <- twsOrder(
#     orderId   = parent_id + 1,
#     action    = "SELL",
#     totalQuantity = qty,
#     orderType = "LMT",
#     lmtPrice = tp_price,
#     parentId = parent_id,
#     transmit = FALSE
# )
# 
# # Stop‑loss order
# sl <- twsOrder(
#     orderId   = parent_id + 2,
#     action    = "SELL",
#     totalQuantity = qty,
#     orderType = "STP",
#     auxPrice = sl_price,
#     parentId = parent_id,
#     transmit = TRUE   # last order transmits the whole bracket
# )
# 
# # Send all three orders
# placeOrder(tws, twsSTK(symbol), parent)
# placeOrder(tws, twsSTK(symbol), tp)
# placeOrder(tws, twsSTK(symbol), sl)

# Define the order processing function
order_function <- function(actions_table, tws) {

    for (i in 1:nrow(actions_table)) {
        
        symbol <- actions_table[i, symbol]
        action <- toupper(actions_table[i, action])
        qty    <- actions_table[i, qty]
        
        # If BUY → create bracket order
        if (action == "BUY") {
            
            tp_price  <- actions_table[i, limit_price]
            sl_price  <- actions_table[i, stop_price]
            
            # Parent order (market buy)
            #parent_id <- reqIds(tws)
            parent_id <- as.numeric(reqIds(tws))
            
            parent <- twsOrder(
                orderId       = parent_id,
                action        = "BUY",
                totalQuantity = qty,
                orderType     = "MKT",
                transmit      = FALSE
            )
            
            # Take‑profit limit order
            tp <- twsOrder(
                orderId       = parent_id + 1,
                action        = "SELL",
                totalQuantity = qty,
                orderType     = "LMT",
                lmtPrice      = tp_price,
                parentId      = parent_id,
                transmit      = FALSE
            )
            
            # Stop‑loss order
            sl <- twsOrder(
                orderId       = parent_id + 2,
                action        = "SELL",
                totalQuantity = qty,
                orderType     = "STP",
                auxPrice      = sl_price,
                parentId      = parent_id,
                transmit      = TRUE   # last order transmits the whole bracket
            )
            
            # Send all three orders
            placeOrder(tws, twsSTK(symbol), parent)
            placeOrder(tws, twsSTK(symbol), tp)
            placeOrder(tws, twsSTK(symbol), sl)
            
        } else {
            
            # Normal SELL order (no bracket)
            placeOrder(
                twsconn  = tws,
                Contract = twsSTK(symbol),
                Order    = twsOrder(reqIds(tws), "SELL", qty, "MKT")
            )
        }
    }
}

# revised version without the loop call:
order_function_dt <- function(actions_table, tws) {
    
    actions_table[, {
        
        symbol <- symbol
        action <- toupper(action)
        qty    <- qty
        
        if (action == "BUY") {
            
            tp_price <- limit_price
            sl_price <- stop_price
            
            parent_id <- as.numeric(reqIds(tws))
            
            parent <- twsOrder(
                orderId       = parent_id,
                action        = "BUY",
                totalQuantity = qty,
                orderType     = "MKT",
                transmit      = FALSE
            )
            
            tp <- twsOrder(
                orderId       = parent_id + 1,
                action        = "SELL",
                totalQuantity = qty,
                orderType     = "LMT",
                lmtPrice      = tp_price,
                parentId      = parent_id,
                transmit      = FALSE
            )
            
            sl <- twsOrder(
                orderId       = parent_id + 2,
                action        = "SELL",
                totalQuantity = qty,
                orderType     = "STP",
                auxPrice      = sl_price,
                parentId      = parent_id,
                transmit      = TRUE
            )
            
            placeOrder(tws, twsSTK(symbol), parent)
            placeOrder(tws, twsSTK(symbol), tp)
            placeOrder(tws, twsSTK(symbol), sl)
            
        } else {
            
            placeOrder(
                twsconn  = tws,
                Contract = twsSTK(symbol),
                Order    = twsOrder(reqIds(tws), "SELL", qty, "MKT")
            )
        }
        
        NULL
    }]
}


# * submit orders THIS IS FOR REAL ----
order_function(actions_table, tws)

# write order history ----
actions_table[, date := today()]

write_rds(actions_table, str_glue("03_actions/{today()}_live_actions_table.rds"))

actions_history_table <- readRDS("03_actions/actions_history_table.rds")

actions_history_table <- rbind(actions_history_table, actions_table, fill = T)

write_rds(actions_history_table, str_glue("03_actions/actions_history_table.rds"))

actions_history_table[,.(pl = sum(unrealizedPNL),value = sum(marketValue)),date][,.(date,pl, value, ROC(value,1))]

# close the tws session ----
twsDisconnect(tws)
