# utils.py
from datetime import datetime

def add_car_age(df):
    df = df.copy()
    df['car_age'] = datetime.now().year - df['make_year']
    return df
