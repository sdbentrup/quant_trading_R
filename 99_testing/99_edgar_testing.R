#tidyedgar testing
# https://gerardgimenezadsuar.github.io/tidyedgar/
library(tidyedgar)

df <- yearly_data(years = 2020:2023)
df

year(from)

net_income <- get_qdata(account = "NetIncomeLoss", years = year(from):year(today()), quarters = c("Q1","Q2","Q3","Q4"))
revenue <- get_qdata(account = "Revenues", years = year(from):year(today()), quarters = c("Q1","Q2","Q3","Q4"))
op_income <- get_qdata(account = "OperatingIncomeLoss", years = year(from):year(today()), quarters = c("Q1","Q2","Q3","Q4"))

prepared <- prepare_data(revenue, net_income, op_income, quarterly = T)
colnames(prepared)

# edgarwebr testing
# https://mwaldstein.github.io/edgarWebR/
# https://github.com/mwaldstein/edgarWebR/issues/22
# remotes::install_github("https://github.com/balthasars/edgarWebR", force = TRUE)
library(edgarWebR)

Sys.setenv(EDGARWEBR_USER_AGENT = "sdbentrup@hotmail.com")
Sys.getenv("EDGARWEBR_USER_AGENT")
Sys.unsetenv("EDGARWEBR_USER_AGENT")

# Safe wrapper function to get CIK
get_company_info <- function(symbol) {
    tryCatch({
        info <- company_information(symbol)
        data.frame(symbol = symbol, 
                   cik_clean = as.numeric(info$cik),
                   cik = info$cik,
                   stringsAsFactors = FALSE)
    }, error = function(e) {
        # Return NA for cik if there's an error
        data.frame(symbol = symbol, cik_clean = NA, cik = NA, stringsAsFactors = FALSE)
    })
}


# unique(prices_base$symbol)

# Example vector of symbols
symbols <- c("BSX","AAPL","FIS","GE","ICE","IRM","LDOS","META","NVDA","TT","VTR","WELL")
symbols_long <- unique(as.character(full_data$symbol))

# Use future_lapply to apply the function in parallel
info_list <- future_lapply(unique(prices_base$symbol), get_company_info)
info_list <- future_lapply(symbols_long, get_company_info)

# Combine the results into a single data.table
info_data <- rbindlist(info_list, fill = TRUE)

# View the result
print(info_data)

setkey(info_data, symbol)

setDT(prepared)

join_edgar <- prepared %>% 
    inner_join(info_data, by = c("data.cik" = "cik_clean")) %>% 
    relocate(symbol, .before = 0) %>% 
    select(-ccp, -taxonomy, -uom, -data.loc) %>% 
    mutate(across(data.start:data.end, ~as.Date(.x))) %>% 
    group_by(symbol) %>% 
    mutate(filing = 1:n()) %>% 
    ungroup()

setDT(join_edgar)

join_edgar[,date := fcase(wday(data.end) == 7, data.end + days(2),
                          wday(data.end) == 1, data.end + days(1),
                          default = data.start)]

join_edgar[,symbol := as.factor(symbol)]

prices_edgar_tbl <- merge(prices_earnings_econ_tbl,
                          join_edgar[,select(.SD,symbol, date, OperatingIncomeLoss:net_margin, filing)])

# still need to fill these down

prices_edgar_tbl <- merge(full_data,
                          join_edgar[,select(.SD,symbol, date, OperatingIncomeLoss:net_margin, filing)],
                          by = c("symbol","date"),
                          all.x = T)
prices_edgar_tbl

# get data from edgarwebR directly
aapl <- company_filings(x = 'AAPL', count = 100)

aapl %>% count(type)

aapl %>% filter(type %in% c('10-K','10-Q')) %>% arrange(accepted_date) %>% pull(href)
parse_filing("https://www.sec.gov/Archives/edgar/data/320193/000032019322000108/0000320193-22-000108-index.htm")

filing <- filing_details("https://www.sec.gov/Archives/edgar/data/320193/000032019322000108/0000320193-22-000108-index.htm")

set_config(user_agent("ScottBentrup sdbentrup@hotmail.com"))

fil <- company_filings("AAPL", type = "10-K", count = 10)

url <- fil$href[1]   # MUST be filing_href, not href
doc <- read_html(url)

doc


# testing edgar package ----
library(edgar)
filings <- edgar::getFilings(320193, filing.year = c(2024:2025), form.type = c('10-K', '10-Q'), useragent = "ScottBentrup sdbentrup@hotmail.com")
edgar::getFilingHeader(320193, filing.year = c(2024:2025), form.type = c('10-K', '10-Q'), useragent = "ScottBentrup sdbentrup@hotmail.com")
headers <- getFilingHeader(320193, filing.year = c(2024:2025), form.type = c('10-K', '10-Q'), useragent = "ScottBentrup sdbentrup@hotmail.com")
items <- get8KItems(320193, filing.year = c(2024:2025), useragent = "ScottBentrup sdbentrup@hotmail.com")

senti.df <- getSentiment(320193, filing.year = c(2024:2025), form.type = c('10-K', '10-Q'),  useragent = "ScottBentrup sdbentrup@hotmail.com")
