library(tidyverse)
library(lubridate)

# GET DATA -------
ds <- read.csv("./Downloads/BSRT Client Numbers.csv", sep = ",")
str(ds)

# DATA FIELDS WRANGLING

# change data fields names
colnames(ds) <- c("date", "locA", "locB")

# add a total column for each day

# modify date field
ds <- ds %>% mutate(
  total = locA + locB, # total demand for each day
  date = as.Date(date, format = "%m/%d/%Y"), # change data format
  month = month(date,abbr = FALSE, label = TRUE), # name of the month
  day_of_week = wday(date, abbr = FALSE, label = TRUE, week_start = 1), # day of the week name
  day_of_month = mday(date), # day of the month number
  season = case_when(
    month %in% c("September", "October", "November") ~ "Autumn",
    month %in%  c("December", "January", "February")  ~ "Winter",
    month %in%  c("March", "April", "May")  ~ "Spring",
    TRUE ~ "Summer") # season
)

# order in ascending time
ds <- ds[order(ds$date),]

# save data
write.csv(ds, file = "./Desktop/Bristol is Open/ODB/homeless_hackthon/souper-duper.csv", row.names = FALSE)




