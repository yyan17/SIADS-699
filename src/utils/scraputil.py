from datetime import timedelta, date

beg_date = date(2008, 1, 1)
# choose the date till which to get the news article urls
end_date = date(2008, 1, 10)

init_date = date(2008, 1, 1)
init_time = 39448

def get_first_date(year, month=1):
    '''
    Provides the first date of an year
    year: year for which first date is inquired
    '''
    first_day = date(year, month, 1)
    return(first_day)

def get_next_date(init_date, init_time):
    '''
    init_date: takes any date
    init_time: numeric code for that date
    returns:returns next date and next init_time code(times of India numeric code for a date)
    '''
    next_date = init_date + timedelta(days = 1)
    init_time += 1
    return(next_date, init_time)

def create_param(next_date, init_time):    
    """
    Return the predicates to access an news article webpage
    :param next_date: date for which to create predicates
    :param init_time: numeric code for that day news articles archive webpage(for every date archive, webpage ends with a numeric value, which is what we are trying to get here to generte the webpage url)
    :return: returns the url predicates
    """
    date_lst = [next_date.year, next_date.month, next_date.day]
    date_str = [str(item) for item in date_lst]
    date_num = [next_date.year, next_date.month, init_time]
    date_param = '/'.join(date_str) 
    PARAMS = ['year-', 'month-', 'starttime-']
    param_str = ''
    for time, param in zip(date_num, PARAMS):
        param_str += param  + str(time) + ','
    param_str = param_str[:-1]
    return(date_param + '/archivelist/' + param_str)

def create_predicates(beg_date, end_date):
    """
    generate predicates for all the dates till end date
    :param beg_date: starting date from which to generate the predicates
    """
#     first_day = get_first_date(beg_year)
    init_time = get_init_time_for_date(beg_date)
    predicates = {}
    next_date = beg_date 
    while next_date <= end_date:
        yield next_date, create_param(next_date, init_time)
        next_date,init_time = get_next_date(next_date, init_time) 
        
def get_init_time_for_date(curr_date):
    # compute the magic number of the given date, which is part of the url of that date news
    date_diff = (curr_date - init_date).days
    curr_init = init_time + date_diff
    return(curr_init)        

