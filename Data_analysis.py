

import re
import pandas as pd

#read the text file
path = "C:/Users/Dell/Desktop/My_Projects/FOR_RESUME/Whatsapp_chat_analysis_2/WhatsApp-Chat-Analysis/WhatsApp Chat with Subhasish ISE.txt"
f = open(path, 'r', encoding='utf-8')
data = f.read()
print(type(data))

#regular expression to find the dates
pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[ap]m\s-\s'

#pass the pattern and data to split it to get the list of messages
messages = re.split(pattern, data)[1:]
print(messages)

#extract all dates
dates = re.findall(pattern, data)

# Remove the trailing " - " and replace non-standard spaces
dates = [date.strip(" -").replace("\u202f", " ") for date in dates]

#create dataframe
df = pd.DataFrame({'user_message': messages, 'message_date': dates})
print(df.head(7))
# convert message_date type
df['message_date'] = pd.to_datetime(df['message_date'],format='%d/%m/%Y, %I:%M %p')
df.rename(columns={'message_date': 'date'}, inplace=True)
df.head(4)