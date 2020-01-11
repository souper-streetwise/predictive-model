# AUTHOR : Elisa Covato
# DATE : 10/11/2019
# DESCRIPTION: We use this script to mpute the missing entries in the
# entries in the weather dataset.

#####################################################################

library(mice)
library(tidyverse)


# GET DATA -------
weather <- read.csv("./data/weather_data.csv", sep = ",", stringsAsFactors = FALSE)
str(weather)


# FIXING PRECIPITATION FIELDS -------

# Precipitation type
# 1. If precip_intensity == 0 then precip_type == no_precip
# We replace "" with no_precip for all these values
for(i in 1:nrow(weather)){
  if(weather$precip_intensity_avg[i] == 0 && 
     weather$precip_intensity_max[i] == 0 &&
     !is.na(weather$precip_intensity_avg[i]) && 
     !is.na(weather$precip_intensity_max[i])) {
    weather$precip_type[i] <- "no_precip"
  }  
}



# Precipitation type
# 2. We replace "" with NA for all the cases for which 
# precipitation is not known
for(i in 1:nrow(weather)){
  if( is.na(weather$precip_intensity_avg[i]) && 
      is.na(weather$precip_intensity_max[i])) {
    weather$precip_type[i] <- NA
  }  
}


# Precipitation type
# 3. Transform values into factors
weather <- weather %>% 
  mutate(
  precip_type = as.factor(precip_type)  
)

# Precipitation average
# Some of the values for precip_intensity_avg are 0 even though 
# precip_intensity_max is not 0. We substitute this with NA
# and we will predict them later.
for(i in 1:nrow(weather)){
  if(weather$precip_intensity_avg[i] == 0 && 
     weather$precip_intensity_max[i] > 0 &&
     !is.na(weather$precip_intensity_avg[i]) && 
     !is.na(weather$precip_intensity_max[i]) ) {
    weather$precip_intensity_avg[i] <-  NA
  } 
}




# COUNTING MISSING DATA -------
weather %>%
  summarise_all(funs(sum(. == "")))
weather %>%
  summarise_all(funs(sum(is.na(.))))




# IMPUTATION -------
# We are now ready to impute the missing data

# 1. Create prediction matrix
# To create the prediction matrix, we need to compute the 
# percentage for each field to be usable in the imputation
percentage_usable_cases <- md.pairs(weather) 
percentage_usable_cases <-round(100*percentage_usable_cases$mr/(percentage_usable_cases$mr+percentage_usable_cases$mm))

# We set to 0 those fields whose percentage is NaN 
# (hence cannot be used for the imputation)
percentage_usable_cases[percentage_usable_cases %in% c("NaN")] <- 0
# We set to 1 those fields whose percentage is not zero
# (hence they will be used for the imputation)
percentage_usable_cases[percentage_usable_cases != 0] <- 1

# Prediction matrix
predM <-  percentage_usable_cases

# Choose imputation method for each variable to impute 
meth <- c(
  "day" = "",               
  "month" = "",
  "year" = "",
  "precip_intensity_max" = "cart",
  "precip_intensity_avg" = "pmm",
  "precip_type" = "cart",
  "wind_speed_max" = "",
  "wind_speed_avg" = "",
  "gust_max" = "cart",
  "gust_avg" = "pmm",
  "temp_min" = "",
  "temp_max" = "",
  "temp_avg"= "",
  "temp_day" = "cart",
  "temp_night" = "cart",
  "humidity" = ""
)


# Imputation
weather_imputed <- weather %>%   mice(
  data = .,
  predictorMatrix = predM,
  method = meth,
  maxit = 0,
  m = 5
)

# Imputation - visualization
stripplot(weather_imputed)
densityplot(weather_imputed) # all imputation
#densityplot(weather_imputed, ~precip_intensity_max)
#densityplot(weather_imputed, ~precip_intensity_avg)
#densityplot(weather_imputed, ~precip_type)
#densityplot(weather_imputed, ~gust_max)
#densityplot(weather_imputed, ~gust_avg)






# Imputation - complete dataset
weather_complete <- mice::complete(weather_imputed)

# SAVE DATA -------
write.csv(weather_complete, "./data/weather_data_complete.csv", row.names = FALSE)

