import pandas as pd
import re

def preprocess(data):
    # Read the file content

    pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[ap]m\s-\s'

    #pass the pattern and data to split it to get the list of messages
    messages = re.split(pattern, data)[1:]

    #extract all dates
    dates = re.findall(pattern, data)

    # Remove the trailing " - " and replace non-standard spaces
    dates = [date.strip(" -").replace("\u202f", " ") for date in dates]

    #create dataframe
    df = pd.DataFrame({'user_message': messages, 'message_date': dates})
    # convert message_date type
    df['message_date'] = pd.to_datetime(df['message_date'],format='%d/%m/%Y, %I:%M %p')
    df.rename(columns={'message_date': 'date'}, inplace=True)

    #separate Users and Message
    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split('([\w\W]+?):\s', message, maxsplit=1)
        if entry[1:]:  # user name
            users.append(entry[1])
            messages.append(" ".join(entry[2:]))
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    #Extract multiple columns from Date Column
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    #print(df[['day_name', 'hour']])

    #extract period
    period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period
    #print(df.head(8))
    return df

#print(preprocess('WhatsApp Chat with Subhasish ISE.txt'))