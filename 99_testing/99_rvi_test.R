# ================================
# Relative Vigor Index (RVI) in R
# ================================

# Load required package
if (!requireNamespace("TTR", quietly = TRUE)) {
    install.packages("TTR")
}
library(TTR)

# Function to calculate RVI
calculate_RVI <- function(open, high, low, close, n = 10, signal_n = 4) {
    # Raw RVI calculation
    rvi_raw <- (close - open) / (high - low)
    
    # Smooth RVI with SMA
    rvi <- SMA(rvi_raw, n = n)
    
    # Signal line (SMA of RVI)
    signal <- SMA(rvi, n = signal_n)
    
    # Return as data frame
    data.frame(
        RVI = rvi,
        Signal = signal
    )
}

# ================================
# Example usage with sample data
# ================================

# Example OHLC data
set.seed(123)
n_points <- 50
open_prices  <- runif(n_points, 100, 110)
high_prices  <- open_prices + runif(n_points, 0.5, 2)
low_prices   <- open_prices - runif(n_points, 0.5, 2)
close_prices <- runif(n_points, low_prices, high_prices)

dt <- data.table(open = open_prices, high = high_prices, low = low_prices, close = close_prices)

dt[,rvi_raw := ((close - open) / (high - low))][,rvi := SMA(rvi_raw, n = 10)][,rvi_signal := SMA(rvi, n = 4)]

# Calculate RVI
rvi_result <- calculate_RVI(open_prices, high_prices, low_prices, close_prices, n = 10, signal_n = 4)

# View last few results
tail(rvi_result)
