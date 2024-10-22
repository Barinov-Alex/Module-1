import numpy as np
import pandas as pd

from prophet import Prophet

import json


if __name__ == '__main__':
    data = pd.read_excel('Данные v2.xlsx', sheet_name='Бр_дневка - 3 (основной)')
    data = data.rename(columns = {'дата': 'ds', 'направление': 'direction', 'выход': 'y'})
    data['ds'] = pd.to_datetime(data['ds'])
    data['direction'] = data['direction'].str.replace('ш', '0').str.replace('л', '1').astype(int)

    data_pred = pd.read_excel('Данные v2.xlsx', sheet_name='Прогноз', usecols=[0], header=0)
    data_pred.columns = ['ds']

    model = Prophet()
    model.fit(data)

    future_dates = model.make_future_dataframe(periods=60)
    forecast = model.predict(future_dates)
    prediction = pd.merge(data_pred, forecast[['ds', 'yhat']], on='ds', how='left')

    prediction.loc[:, 'direction'] = (prediction['yhat'].diff() > 0).astype(int)

    prediction.loc[prediction['yhat'].isna()]

    prediction['direction'].to_json('forecast_class.json', orient='records')