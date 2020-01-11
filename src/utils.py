def get_path(folder: str):
    from pathlib import Path
    return Path.cwd().parent / folder

def date_iter(start_date, end_date) -> iter:
    from datetime import timedelta
    curr_date = start_date
    while curr_date <= end_date:
        yield curr_date
        curr_date += timedelta(days = 1)

def get_dates(start_date, end_date) -> list:
    from datetime import timedelta
    return [date for date in date_iter(start_date, end_date)]

def preciptype2int(precip_type: str):
    if precip_type == 'no_precip': return 0
    elif precip_type == 'rain': return 1
    elif precip_type == 'snow': return 2
    elif precip_type == 'sleet': return 3

if __name__ == '__main__':
    pass
