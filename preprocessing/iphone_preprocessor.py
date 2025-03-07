

import re
import pandas as pd

def iphone_preprocessor(data):
    separator_pattern = r'\[\d{1,2}/\d{1,2}/\d{2,4},\s(?:[01]?\d|2[0-3]):[0-5]\d:[0-5]\d\s?[APap][Mm]\]'

    split_messages = re.split(separator_pattern, data)[1:]
    text_dates = re.findall(separator_pattern, data)

    dataframes = pd.DataFrame({'user_message': split_messages, 'message_date': text_dates})
    dataframes['message_date'] = pd.to_datetime(dataframes['message_date'], format='[%d/%m/%Y, %I:%M:%S %p]')
    dataframes.rename(columns={'message_date': 'date'}, inplace=True)

    users = []
    messages = []
    # Iterate through the 'user_message' column which exists in the DataFrame
    for message in dataframes['user_message']:
        entry = re.split('([\w\W]+?):\s', message)
        if len(entry) > 1:  # Valid message with a user
            users.append(entry[1])
            messages.append(" ".join(entry[2:]))
        else:
            # Handle system messages like group notifications
            users.append('group_notification')
            messages.append(entry[0])

    dataframes['user'] = users
    dataframes['message'] = messages

    # Extracting date parts
    dataframes['only_date'] = dataframes['date'].dt.date
    dataframes['year'] = dataframes['date'].dt.year
    dataframes['month_num'] = dataframes['date'].dt.month
    dataframes['month'] = dataframes['date'].dt.month_name()
    dataframes['day'] = dataframes['date'].dt.day
    dataframes['day_name'] = dataframes['date'].dt.day_name()
    dataframes['hour'] = dataframes['date'].dt.hour
    dataframes['minute'] = dataframes['date'].dt.minute

    # Add a 'period' column for time range analysis
    period = []
    for hour in dataframes[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    dataframes['period'] = period

    return dataframes