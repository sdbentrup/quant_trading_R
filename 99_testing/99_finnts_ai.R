library(finnts)

browseVignettes("finnts")

hist_data <- timetk::m4_monthly %>%
    dplyr::filter(date >= "2013-01-01") %>%
    dplyr::rename(Date = date) %>%
    dplyr::mutate(id = as.character(id))

print(hist_data)

print(unique(hist_data$id))

# connect to LLM via Azure AI
driver_llm <- ellmer::chat_azure_openai(model = "gpt-4o-mini")

# set up new forecast project and agent run
project <- set_project_info(
    project_name = "Demo_Project",
    combo_variables = c("id"),
    target_variable = "value",
    date_type = "month"
)

agent <- set_agent_info(
    project_info = project,
    driver_llm = driver_llm,
    input_data = hist_data,
    forecast_horizon = 12,
    hist_end_date = as.Date("2014-12-01")
)

iterate_forecast(
    agent_info = agent,
    max_iter = 3,
    weighted_mape_goal = 0.03
)

forecast_output <- get_agent_forecast(agent_info = agent)

agent_run_results <- get_best_agent_run(agent_info = agent, full_run_info = TRUE)

# set up agent with updated data
# overwrite creates a new version of the agent, which is required when running update_forecast()
agent <- set_agent_info(
    project_info = project,
    driver_llm = driver_llm,
    input_data = hist_data,
    forecast_horizon = 6,
    hist_end_date = as.Date("2015-06-01"),
    overwrite = TRUE
)

# update forecast
update_forecast(
    agent_info = agent,
    weighted_mape_goal = 0.03
)

# get updated forecast output
updated_forecast_output <- get_agent_forecast(agent_info = agent)


