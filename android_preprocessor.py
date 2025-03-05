

import re
import pandas as pd

def android_preprocessor(data):
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s(?:[01]?\d|2[0-3]):[0-5]\d\s?[APap][Mm]?\s-\s'

    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    df = pd.DataFrame({'message_date': dates, 'user_message': messages})

    # Clean the message_date column
    df['message_date'] = df['message_date']
    df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%Y, %I:%M %p - ')
    df.rename(columns={'message_date': 'date'}, inplace=True)

    # Create 'user' and 'message' columns
    users = []
    message_contents = []
    for message in df['user_message']:
        entry = re.split('([\w\W]+?):\s', message)
        if len(entry) > 2:
            users.append(entry[1])
            message_contents.append(entry[2])
        else:
            users.append('group_notification')
            message_contents.append(entry[0])

    df['user'] = users
    df['message'] = message_contents
    df.drop(columns=['user_message'], inplace=True)

    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute


    period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period


    return df
