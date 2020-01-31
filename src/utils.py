def boolean(input):
    ''' Convert strings 'true'/'false' into boolean True/False.
    INPUT
        input: str or bool
    OUTPUT
        A bool object which is True if input is 'true' and False 
        if input is 'false' (not case sensitive). If input is already
        of type bool then nothing happens, and if none of the above
        conditions are true then a None object is returned.
    '''
    if isinstance(input, bool): return input
    if isinstance(input, str) and input.lower() == 'true': return True
    if isinstance(input, str) and input.lower() == 'false': return False

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

def load_model_data(model_name: str = 'soup_model', data_dir: str = 'data'):
    ''' Load a machine learning model. '''
    import pickle
    model_path = get_path(data_dir) / model_name
    if model_path.is_file():
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f'The model {model_path} was not found.')

class TQDM(object):
    ''' TQDM class to be used in Bayesian optimisation with skopt. '''

    def __init__(self, update_amount: int = 1, **kwargs):
        from tqdm.auto import tqdm
        self.bar = tqdm(**kwargs)
        self.update_amount = update_amount

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def __call__(self, x):
        self.bar.update(self.update_amount)

    def close(self):
        self.bar.close()


if __name__ == '__main__':
    pass
