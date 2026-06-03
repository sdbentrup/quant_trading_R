library(tidyverse)
library(data.table)
library(plotly)

# testing fmp api ----
# this is a different package to the other fmpapi on cran. see below
# https://jpiburn.github.io/fmpapi/
# seems to use a legacy api that is broken

#remotes::install_github('jpiburn/fmpapi')
library(fmpapi)

api_key <- "a20f7e263ca3ca1504945c034251a638"
fmp_api_key(api_key)

# reload
readRenviron('~/.Renviron')

my_stocks <- c("AAPL", "GE")

d <- fmp_profile(my_stocks)

glimpse(d)


dcf <- fmp_dcf(my_stocks, historical = TRUE, quarterly = TRUE) 

dcf %>%
    ggplot(
        aes(x = date, y = dcf, colour = symbol)
    ) +
    geom_line(size = 1) +
    theme_minimal() +
    labs(
        title = "Historical Discounted Cash Flow Valuation",
        y = "Discounted Cash Flow Value"
    )


# tidyfinance implementation ----
# https://tidy-finance.github.io/r-fmpapi/
library(fmpapi)

fmp_set_api_key()

fmp_get(resource = "profile", symbol = "AAPL")

fmp_get(resource = "balance-sheet-statement", symbol = "AAPL", params = list(period = "annual", limit = 5))

bal <- fmp_get(resource = "balance-sheet-statement", symbol = "AAPL", params = list(period = "quarter", limit = 5))

fmp_get(resource = "income-statement", symbol = "AAPL")

# Get available balance sheet statements
fmp_get(
    resource = "balance-sheet-statement",
    symbol = "AAPL"
)

# Get last income statements
fmp_get(
    resource = "income-statement",
    symbol = "AAPL",
    params = list(limit = 1)
)

# Get cash flow statements
cf_q <- fmp_get(
    resource = "cash-flow-statement",
    symbol = "AAPL",
    params = list(period = "quarter")
)

cf_a <- fmp_get(
    resource = "cash-flow-statement",
    symbol = "AAPL",
    params = list(period = "annual", limit = 5)
)

# Get historical market capitalization
fmp_get(
    resource = "historical-market-capitalization",
    symbol = "UNH",
    params = list(from = "2023-12-01", to = "2023-12-31")
)

# Get stock list
fmp_get(
    resource = "stock/list"
)

# Get company profile
profile <- fmp_get(
    resource = "profile", symbol = "AAPL"
)

# Search for stock information
fmp_get(
    resource = "search", params = list(query = "AAPL")
)


# Get data with original column names
fmp_get(
    resource = "profile", symbol = "AAPL", snake_case = FALSE
)

fmp_get(
    resource = "discounted-cash-flow", symbol = "AAPL"
)

grades <- fmp_get(
    resource = "grades-historical", symbol = "AAPL"
) # can only do one at a time


fmp_get(
    resource = "historical-discounted-cash-flow", symbol = "AAPL",  params = list(period = "quarter", limit = 5))

# * financial statements analysis ----
# https://www.tidy-finance.org/r/financial-statement-analysis.html

library(scales)
library(ggrepel)

sample <- c(
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "DIS", "NKE",
    "WMT", "KO", "JPM", "BAC", "V", "XOM", "CVX", "JNJ", "PFE", "INTC",
    "AMD", "SBUX", "BABA", "UBER", "CSCO"
)

params <- list(period = "annual", limit = 5)

balance_sheet_statements <- sample |>
    map_df(
        \(x) {
            fmp_get(resource = "balance-sheet-statement", symbol = x, params = params)
        }
    )

income_statements <- sample |>
    map_df(
        \(x) fmp_get(resource = "income-statement", symbol = x, params = params)
    )

cash_flow_statements <- sample |>
    map_df(
        \(x) fmp_get(resource = "cash-flow-statement", symbol = x, params = params)
    )

selected_symbols <- c("MSFT", "AAPL", "AMZN", "NVDA")

balance_sheet_statements <- balance_sheet_statements |>
    mutate(
        fiscal_year = as.integer(fiscal_year),
        current_ratio = total_current_assets / total_assets,
        quick_ratio = (total_current_assets - inventory) /
            total_current_liabilities,
        cash_ratio = cash_and_cash_equivalents / total_current_liabilities,
        label = if_else(symbol %in% selected_symbols, symbol, NA),
    )

balance_sheet_statements |>
    filter(fiscal_year == 2023 & !is.na(label)) |>
    select(symbol, contains("ratio")) |>
    pivot_longer(-symbol) |>
    mutate(name = str_to_title(str_replace_all(name, "_", " "))) |>
    ggplot(aes(x = value, y = name, fill = symbol)) +
    geom_col(position = "dodge") +
    scale_x_continuous(labels = percent) +
    labs(
        x = NULL,
        y = NULL,
        fill = NULL,
        title = "Liquidity ratios for selected stocks for 2023"
    )

balance_sheet_statements <- balance_sheet_statements |>
    mutate(
        debt_to_equity = total_debt / total_equity,
        debt_to_asset = total_debt / total_assets
    )

income_statements <- income_statements |>
    mutate(
        fiscal_year = as.integer(fiscal_year),
        interest_coverage = operating_income / interest_expense,
        #label = if_else(symbol %in% selected_symbols, symbol, NA),
    )

balance_sheet_statements |>
    filter(symbol %in% selected_symbols) |>
    ggplot(aes(x = fiscal_year, y = debt_to_asset, color = symbol)) +
    geom_line(linewidth = 1) +
    scale_y_continuous(labels = percent) +
    labs(
        x = NULL,
        y = NULL,
        color = NULL,
        title = "Debt-to-asset ratios of selected stocks between 2020 and 2024"
    )

selected_colors <- c("#F21A00", "#EBCC2A", "#78B7C5", "#3B9AB2", "lightgrey")

balance_sheet_statements |>
    filter(fiscal_year == 2023) |>
    ggplot(
        aes(x = debt_to_asset, y = fct_reorder(symbol, debt_to_asset), fill = label)
    ) +
    geom_col() +
    scale_x_continuous(labels = percent) +
    scale_fill_manual(values = selected_colors) +
    labs(
        x = NULL,
        y = NULL,
        color = NULL,
        title = "Debt-to-asset ratios of selected stocks in 2023"
    ) +
    theme(legend.position = "none")

income_statements |>
    filter(fiscal_year == 2023) |>
    select(symbol, interest_coverage, fiscal_year) |>
    left_join(
        balance_sheet_statements,
        join_by(symbol, fiscal_year)
    ) |>
    ggplot(aes(x = debt_to_asset, y = interest_coverage, color = label)) +
    geom_point(size = 2) +
    geom_label_repel(aes(label = label), seed = 42, box.padding = 0.75) +
    scale_x_continuous(labels = percent) +
    scale_y_continuous(labels = percent) +
    scale_color_manual(values = selected_colors) +
    labs(
        x = "Debt-to-Asset",
        y = "Interest Coverage",
        title = "Debt-to-asset ratios and interest coverages for selected stocks"
    ) +
    theme(legend.position = "none")

combined_statements <- balance_sheet_statements |>
    select(
        symbol,
        fiscal_year,
        label,
        current_ratio,
        quick_ratio,
        cash_ratio,
        debt_to_equity,
        debt_to_asset,
        total_assets,
        total_equity
    ) |>
    left_join(
        income_statements |>
            select(
                symbol,
                fiscal_year,
                interest_coverage,
                revenue,
                cost_of_revenue,
                selling_general_and_administrative_expenses,
                interest_expense,
                gross_profit,
                net_income
            ),
        join_by(symbol, fiscal_year)
    ) |>
    left_join(
        cash_flow_statements |>
            mutate(fiscal_year = as.integer(fiscal_year)) |> 
            select(symbol, fiscal_year, inventory, accounts_receivables),
        join_by(symbol, fiscal_year)
    )

combined_statements <- combined_statements |>
    mutate(
        asset_turnover = revenue / total_assets,
        inventory_turnover = cost_of_revenue / inventory,
        receivables_turnover = revenue / accounts_receivables
    )

combined_statements <- combined_statements |>
    mutate(
        gross_margin = gross_profit / revenue,
        profit_margin = net_income / revenue,
        after_tax_roe = net_income / total_equity
    )

combined_statements |>
    filter(symbol %in% selected_symbols) |>
    ggplot(aes(x = fiscal_year, y = gross_margin, color = symbol)) +
    geom_line() +
    scale_y_continuous(labels = percent) +
    labs(
        x = NULL,
        y = NULL,
        color = NULL,
        title = "Gross margins for selected stocks between 2019 and 2023"
    )

combined_statements |>
    filter(fiscal_year == 2023) |>
    ggplot(aes(x = gross_margin, y = profit_margin, color = label)) +
    geom_point(size = 2) +
    geom_label_repel(aes(label = label), seed = 42, box.padding = 0.75) +
    scale_x_continuous(labels = percent) +
    scale_y_continuous(labels = percent) +
    scale_color_manual(values = selected_colors) +
    labs(
        x = "Gross margin",
        y = "Profit margin",
        title = "Gross and profit margins for selected stocks in 2023"
    ) +
    theme(legend.position = "none")

financial_ratios <- combined_statements |>
    filter(fiscal_year == 2023) |>
    select(
        symbol,
        contains(c(
            "ratio",
            "margin",
            "roe",
            "_to_",
            "turnover",
            "interest_coverage"
        ))
    ) |>
    pivot_longer(cols = -symbol) |>
    mutate(
        type = case_when(
            name %in% c("current_ratio", "quick_ratio", "cash_ratio") ~
                "Liquidity Ratios",
            name %in% c("debt_to_equity", "debt_to_asset", "interest_coverage") ~
                "Leverage Ratios",
            name %in%
                c("asset_turnover", "inventory_turnover", "receivables_turnover") ~
                "Efficiency Ratios",
            name %in% c("gross_margin", "profit_margin", "after_tax_roe") ~
                "Profitability Ratios"
        )
    )

financial_ratios |>
    group_by(type, name) |>
    arrange(desc(value)) |>
    mutate(rank = row_number()) |>
    group_by(symbol, type) |>
    summarize(rank = mean(rank), .groups = "drop") |>
    filter(symbol %in% selected_symbols) |>
    ggplot(aes(x = rank, y = type, color = symbol)) +
    geom_point(shape = 17, size = 4) +
    scale_color_manual(values = selected_colors) +
    labs(
        x = "Average rank",
        y = NULL,
        color = NULL,
        title = "Average rank among selected stocks"
    ) +
    coord_cartesian(xlim = c(1, 30))

market_cap <- sample |>
    map_df(
        \(x) {
            fmp_get(
                resource = "historical-market-capitalization",
                x
            )
        }
    ) |> 
    filter(date == min(date))

combined_statements_ff <- combined_statements |>
    filter(fiscal_year == 2023) |>
    left_join(market_cap, join_by(symbol)) |>
    left_join(
        balance_sheet_statements |>
            filter(fiscal_year == 2022) |>
            select(symbol, total_assets_lag = total_assets),
        join_by(symbol)
    ) |>
    mutate(
        size = log(market_cap),
        book_to_market = total_equity / market_cap,
        operating_profitability = (revenue -
                                       cost_of_revenue -
                                       selling_general_and_administrative_expenses -
                                       interest_expense) /
            total_equity,
        investment = total_assets / total_assets_lag
    )

combined_statements_ff |>
    select(
        symbol,
        Size = size,
        `Book-to-Market` = book_to_market,
        `Profitability` = operating_profitability,
        Investment = investment
    ) |>
    pivot_longer(-symbol) |>
    group_by(name) |>
    arrange(desc(value)) |>
    mutate(rank = row_number()) |>
    ungroup() |>
    #filter(symbol %in% selected_symbols) |>
    ggplot(aes(x = rank, y = name, color = symbol)) +
    geom_point(shape = 17, size = 4) +
    #scale_color_manual(values = selected_colors) +
    labs(
        x = "Rank",
        y = NULL,
        color = NULL,
        title = "Rank in Fama-French variables for selected stocks"
    ) +
    coord_cartesian(xlim = c(1, 30))

# * dcf analysis ----
# https://www.tidy-finance.org/r/discounted-cash-flow-analysis.html
symbol <- "MSFT"

income_statements <- fmp_get(
    "income-statement",
    symbol,
    list(period = "annual", limit = 5)
)

cash_flow_statements <- fmp_get(
    "cash-flow-statement",
    symbol,
    list(period = "annual", limit = 5)
)

dcf_data <- income_statements |>
    mutate(
        fiscal_year = as.integer(fiscal_year),
        ebit = net_income + income_tax_expense + interest_expense - interest_income
    ) |>
    select(
        year = fiscal_year,
        ebit,
        revenue,
        depreciation_and_amortization,
        taxes = income_tax_expense
    ) |>
    left_join(
        cash_flow_statements |>
            mutate(
                fiscal_year = as.integer(fiscal_year)
            ) |>
            select(
                year = fiscal_year,
                delta_working_capital = change_in_working_capital,
                capex = capital_expenditure
            ),
        join_by(year)
    ) |>
    mutate(
        fcf = ebit +
            depreciation_and_amortization -
            taxes +
            delta_working_capital -
            capex
    ) |>
    arrange(year)

dcf_data <- dcf_data |>
    mutate(
        revenue_growth = revenue / lag(revenue) - 1,
        operating_margin = ebit / revenue,
        da_margin = depreciation_and_amortization / revenue,
        taxes_to_revenue = taxes / revenue,
        delta_working_capital_to_revenue = delta_working_capital / revenue,
        capex_to_revenue = capex / revenue
    )
library(scales)
dcf_data |>
    pivot_longer(cols = c(operating_margin:capex_to_revenue)) |>
    ggplot(aes(x = year, y = value, color = name)) +
    geom_line() +
    scale_x_continuous(breaks = pretty_breaks()) +
    scale_y_continuous(labels = percent) +
    labs(
        x = NULL,
        y = NULL,
        color = NULL,
        title = "Key financial ratios of Microsoft between 2021 and 2025"
    )

compute_terminal_value <- function(last_fcf, growth_rate, discount_rate) {
    last_fcf * (1 + growth_rate) / (discount_rate - growth_rate)
}

last_fcf <- tail(dcf_data$fcf, 1)
terminal_value <- compute_terminal_value(last_fcf, 0.04, 0.08)
terminal_value / 1e9

compute_dcf <- function(wacc, growth_rate) {
    free_cash_flow <- dcf_data$fcf
    last_fcf <- tail(free_cash_flow, 1)
    terminal_value <- compute_terminal_value(last_fcf, growth_rate, wacc)
    
    years <- length(free_cash_flow)
    present_value_fcf <- free_cash_flow / (1 + wacc)^(1:years)
    present_value_tv <- terminal_value / (1 + wacc)^years
    total_dcf_value <- sum(present_value_fcf) + present_value_tv
    total_dcf_value
}

compute_dcf(0.1, 0.04) / 1e9

wacc_range <- seq(0.06, 0.12, by = 0.01)
growth_rate_range <- seq(0.02, 0.05, by = 0.01)

sensitivity <- expand_grid(
    wacc = wacc_range,
    growth_rate = growth_rate_range
) |>
    mutate(value = pmap_dbl(list(wacc, growth_rate), compute_dcf))

sensitivity |>
    mutate(value = round(value / 1e9, 0)) |>
    ggplot(aes(x = wacc, y = growth_rate, fill = value)) +
    geom_tile() +
    geom_text(aes(label = comma(value)), color = "white") +
    scale_x_continuous(labels = percent) +
    scale_y_continuous(labels = percent) +
    scale_fill_continuous(labels = comma) +
    labs(
        x = "WACC",
        y = "Perpetual growth rate",
        fill = "Enterprise value",
        title = "Enterprise value of Microsoft for different WACC and growth rate scenarios"
    ) +
    guides(fill = guide_colorbar(barwidth = 15, barheight = 0.5))

# indicators for stocks ----
library(timetk)


# * analyst ratings ----
grades_test <- map_dfr(sample,
                       ~fmp_get(resource = "grades-historical", symbol = .x))

setDT(grades_test)
grades_test[, buy_share := (analyst_ratings_strong_buy + analyst_ratings_buy)/rowSums(.SD),.SDcols = patterns("ratings")]

grades %>% 
    pivot_longer(-(symbol:date)) %>% 
    ggplot(aes(x = date, y = value, color = name))+
    geom_line(alpha= 0.7)

ggplotly(grades_test %>% 
             pivot_longer(-(symbol:date)) %>% 
             ggplot(aes(x = date, y = value, color = symbol))+
             geom_line(alpha= 0.7, show.legend = F)+
             facet_wrap(~name, scales = "free_y")+
             theme_clean(base_family = "Arial"))

grades_test_pad <- grades_test %>% 
    group_by(symbol) %>% 
    pad_by_time(date, .fill_na_direction = "down", .end_date = today()) %>% 
    mutate(change_buy = (analyst_ratings_strong_buy+analyst_ratings_buy)-data.table::shift(analyst_ratings_strong_buy+analyst_ratings_buy,1),
           change_hold = (analyst_ratings_hold)-data.table::shift(analyst_ratings_hold,1),
           change_sell = (analyst_ratings_strong_sell+analyst_ratings_sell)-data.table::shift(analyst_ratings_strong_sell+analyst_ratings_sell,1)) %>% 
    ungroup()

ggplotly(grades_test_pad %>% 
             pivot_longer(-(symbol:date)) %>% 
             ggplot(aes(x = date, y = value, color = symbol))+
             geom_line(alpha= 0.7, show.legend = F)+
             facet_wrap(~name, scales = "free_y")+
             theme_clean(base_family = "Arial"))

full_data <- merge(full_data[symbol %in% sample],
                           grades_test_pad,
                           by = c("symbol","date"))

# * dcf ----
# (historical - maybe need to pay for this?)
# * financial ratios ----
# base data 

params <- list(period = "annual", limit = 5)
params <- list(period = "annual", limit = 5)

sample <- c(
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "DIS", "NKE",
    "WMT", "KO", "JPM", "BAC", "V", "XOM", "CVX", "JNJ", "PFE", "INTC",
    "AMD", "SBUX", "BABA", "UBER", "CSCO"
)

balance_sheet_statements <- sample |>
    map_df(
        \(x) {
            fmp_get(resource = "balance-sheet-statement", symbol = x, params = params)
        }
    )

income_statements <- sample |>
    map_df(
        \(x) fmp_get(resource = "income-statement", symbol = x, params = params)
    )

cash_flow_statements <- sample |>
    map_df(
        \(x) fmp_get(resource = "cash-flow-statement", symbol = x, params = params)
    )

# liquidity ----
# c("current_ratio", "quick_ratio", "cash_ratio")

balance_sheet_statements <- balance_sheet_statements |>
    mutate(
        fiscal_year = as.integer(fiscal_year),
        current_ratio = total_current_assets / total_assets,
        quick_ratio = (total_current_assets - inventory) /
            total_current_liabilities,
        cash_ratio = cash_and_cash_equivalents / total_current_liabilities,
        #label = if_else(symbol %in% selected_symbols, symbol, NA),
    )

# leverage ----
# c("debt_to_equity", "debt_to_asset", "interest_coverage")

balance_sheet_statements <- balance_sheet_statements |>
    mutate(
        debt_to_equity = total_debt / total_equity,
        debt_to_asset = total_debt / total_assets
    )

income_statements <- income_statements |>
    mutate(
        fiscal_year = as.integer(fiscal_year),
        interest_coverage = operating_income / interest_expense,
        #label = if_else(symbol %in% selected_symbols, symbol, NA),
    )

combined_statements <- balance_sheet_statements |>
    select(
        symbol,
        fiscal_year,
        #label,
        current_ratio,
        quick_ratio,
        cash_ratio,
        debt_to_equity,
        debt_to_asset,
        total_assets,
        total_equity
    ) |>
    left_join(
        income_statements |>
            select(
                symbol,
                fiscal_year,
                interest_coverage,
                revenue,
                cost_of_revenue,
                selling_general_and_administrative_expenses,
                interest_expense,
                gross_profit,
                net_income
            ),
        join_by(symbol, fiscal_year)
    ) |>
    left_join(
        cash_flow_statements |>
            mutate(fiscal_year = as.integer(fiscal_year)) |> 
            select(symbol, fiscal_year, inventory, accounts_receivables),
        join_by(symbol, fiscal_year)
    )

# efficiency ----
# c("asset_turnover", "inventory_turnover", "receivables_turnover")

combined_statements <- combined_statements |>
    mutate(
        asset_turnover = revenue / total_assets,
        inventory_turnover = cost_of_revenue / inventory,
        receivables_turnover = revenue / accounts_receivables
    )

# profitability ----
# c("gross_margin", "profit_margin", "after_tax_roe")

combined_statements <- combined_statements |>
    mutate(
        gross_margin = gross_profit / revenue,
        profit_margin = net_income / revenue,
        after_tax_roe = net_income / total_equity
    )

# fama french ratio ----
market_cap <- sample |>
    map_df(
        \(x) {
            fmp_get(
                resource = "historical-market-capitalization",
                x
            )
        }
    ) |> 
    filter(date == min(date))

# test this further
combined_statements_ff <- combined_statements |>
    #filter(fiscal_year == 2023) |>
    left_join(market_cap, join_by(symbol)) |>
    mutate(
        size = log(market_cap),
        book_to_market = total_equity / market_cap,
        operating_profitability = (revenue -
                                       cost_of_revenue -
                                       selling_general_and_administrative_expenses -
                                       interest_expense) /
            total_equity,
        investment = total_assets / shift(total_assets, -1)
    )

combined_statements_ff %>% select(symbol,fiscal_year, size, book_to_market, operating_profitability, investment)

combined_statements_ff %>% 
    select(symbol,
           current_ratio, quick_ratio, cash_ratio,
           debt_to_equity, debt_to_asset, interest_coverage,
           asset_turnover, inventory_turnover, receivables_turnover,
           gross_margin, profit_margin, after_tax_roe,
           size, book_to_market, operating_profitability, investment)


# changes in ratios?
