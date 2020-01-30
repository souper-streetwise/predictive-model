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

def precip_type(inputs):
    precips = ['no_precip', 'rain', 'snow', 'sleet']
    is_idx = isinstance(inputs, int)
    return precips[inputs] if is_idx else precips.index(inputs)

def day_of_week(inputs):
    ''' Convert between numbers 1-7 and days of the week. '''
    days_of_week = ['monday', 'tuesday', 'wednesday', 'thursday',
                'friday', 'saturday', 'sunday']
    is_idx = isinstance(inputs, int)
    return days_of_week[inputs-1] if is_idx else days_of_week.index(inputs)+1

def month(inputs):
    ''' Convert between numbers 1-12 and months. '''
    months = ['january', 'february', 'march', 'april', 'may', 'june',
              'july', 'august', 'september', 'october', 'november', 'december']
    is_idx = isinstance(inputs, int)
    return months[inputs - 1] if is_idx else months.index(inputs) + 1

def load_model_data(data_dir: str = 'data'):
    ''' Load a machine learning model. '''
    from utils import get_path
    import pickle
    model_path = get_path(data_dir) / 'extra_trees'
    if model_path.is_file():
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f'No model found in {data_dir}.')


if __name__ == '__main__':
    pass
