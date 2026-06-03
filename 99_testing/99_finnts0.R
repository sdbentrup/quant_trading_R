## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  warning = FALSE,
  message = FALSE
)
library(finnts)

## ----message = FALSE, eval = FALSE--------------------------------------------
#  library(finnts)
#  
#  browseVignettes("finnts")

## ----message = FALSE----------------------------------------------------------
library(finnts)

hist_data <- timetk::m4_monthly %>%
  dplyr::filter(date >= "2013-01-01") %>%
  dplyr::rename(Date = date) %>%
  dplyr::mutate(id = as.character(id))

print(hist_data)

print(unique(hist_data$id))

## ----message = FALSE, eval = hist_data, error=FALSE, warning = FALSE, echo=T, eval = TRUE----
run_info <- set_run_info(
  experiment_name = "finn_forecast",
  run_name = "test_run"
)

## ----message = FALSE, eval = hist_data, error=FALSE, warning = FALSE, echo=T, eval = TRUE----
# no need to assign it to a variable, since all of the outputs are written to disk :)
forecast_time_series(
  run_info = run_info,
  input_data = hist_data,
  combo_variables = c("id"),
  target_variable = "value",
  date_type = "month",
  forecast_horizon = 3,
  back_test_scenarios = 6,
  models_to_run = c("arima", "ets"),
  return_data = FALSE
)

## ----message = FALSE, eval = finn_output, message = FALSE, eval = FALSE, echo=T----
 finn_output_tbl <- get_forecast_data(run_info = run_info)

 print(finn_output_tbl)

## ----message = FALSE, eval = finn_output, message = FALSE, eval = FALSE, echo=T----
 future_forecast_tbl <- finn_output_tbl %>%
   dplyr::filter(Run_Type == "Future_Forecast")

 print(future_forecast_tbl)

## ----message = FALSE, eval = finn_output, eval = FALSE, echo=T----------------
 back_test_tbl <- finn_output_tbl %>%
   dplyr::filter(Run_Type == "Back_Test")

 print(back_test_tbl)

## ----message = FALSE, eval = finn_output, eval = FALSE, echo=T----------------
 best_model_tbl <- finn_output_tbl %>%
   dplyr::filter(Best_Model == "Yes") %>%
   dplyr::select(Combo, Model_ID, Model_Name, Model_Type, Recipe_ID) %>%
   dplyr::distinct()

 print(best_model_tbl)

## ----message = FALSE, eval = finn_output, eval = FALSE, echo=T----------------
 trained_model_tbl <- get_trained_models(run_info = run_info)

 print(trained_model_tbl)

## ----message = FALSE, eval = finn_output, eval = FALSE, echo=T----------------
 R1_prepped_data_tbl <- get_prepped_data(
   run_info = run_info,
   recipe = "R1"
 )

 print(R1_prepped_data_tbl)

 R2_prepped_data_tbl <- get_prepped_data(
   run_info = run_info,
   recipe = "R2"
 )

 print(R2_prepped_data_tbl)

## ----message = FALSE, eval = finn_output, eval = FALSE, echo=T----------------
 run_info_tbl <- get_run_info(experiment_name = "finn_forecast")

 print(run_info_tbl)

