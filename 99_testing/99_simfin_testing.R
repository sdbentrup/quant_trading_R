# SimFin testing
library(simfinapi)
sfa_set_api_key("e8b8347f-de1f-4373-8a12-db846554f759")

tickers <- c("AMZN", "GOOG") # Amazon, Google
prices <- sfa_load_shareprices(tickers)

# load ggplot2
library(ggplot2)

# create plot
ggplot(prices) +
    aes(x = Date, y = `Last Closing Price`, color = name) +
    geom_line()

statements <- sfa_load_statements(tickers, statements = c('pl','bs','cf'))

tickers_long <- unique(full_data$symbol) %>% as.character()
full_statements <- sfa_load_statements(tickers_long, statements = c('pl'))
