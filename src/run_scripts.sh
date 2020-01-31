#!/bin/sh
python training.py --model_name soup_vanilla
python training.py --model_name soup_no_month --include_month false
python training.py --model_name soup_no_month_day_of_month --include_month false --include_day_of_month false
python training.py --model_name soup_no_date_info --include_month false --include_day_of_month false --include_day_of_week false
