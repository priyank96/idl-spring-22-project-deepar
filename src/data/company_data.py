'''
Process raw company data json and transform to a format that a RNN can consume.
Operations performed:
1. Discard files that have no data in them (hit the API rate limit)
   The contents of the file will be like: 
   {"Information": "Thank you for using Alpha Vantage! You have reached the 500 requests/day limit .. }
   {"Error Message": ...}
2. Well formed JSON files have the following contents:
    {
  ...
  "Time Series (Daily)": {
    "2022-02-07": {
      "1. open": "141.5000",
      "2. high": "142.4900",
      "3. low": "139.8100",
      "4. close": "140.3700",
      "5. volume": "1263668"
    },
    "2022-02-04": {
      "1. open": "141.4800",
      "2. high": "142.3450",
      "3. low": "139.6700",
      "4. close": "141.1200",
      "5. volume": "1087398"
    },
  }

  Each such file is transformed into a pandas dataframe.
  On this data frame we apply the following operations:
    1. Split timestamp into month, day columns
    2. Drop year column
    3. Normalize each column
    4. Iterate over the rows of this dataframe with a window size of n, moving down one row at a time.
    5. For each window make two numpy matrices of shape (n-1 x ):
        1. Input: row 0 to n-1 with [month,day,open,low,high,close, company name]
        2. Output: row 1 to n [close]
'''
from pathlib import Path
import sys
import os
import json 
import tqdm
import pandas as pd
import numpy as np
from scipy.stats import zscore

sys.path.insert(0,str(Path(__file__).parent.parent))
from constants import API_COMPANY_DATA_PATH, WINDOW_SIZE, DATA_PATH

all_inputs = None
all_labels = None


def discard_for_api_error(data):
    if "Information" in data:
        return True
    if "Error Message" in data:
        return True
    return False

def make_data_frame(data, stock_name):
    rows = []
    for day_data_key in data["Time Series (Daily)"]:
        day_data = data["Time Series (Daily)"][day_data_key]
        rows.append([
            day_data_key,
            float(day_data["1. open"]), 
            float(day_data["2. high"]),
            float(day_data["3. low"]),
            float(day_data["4. close"]),
            float(day_data["5. volume"])
        ])
    df = pd.DataFrame(rows, columns=["date","open","high","low","close","volume"])
    df["date"] = pd.to_datetime(df["date"],infer_datetime_format=True)
    df = df.sort_values(by="date")
    df['day'] = [d.day for d in df['date']]
    df['month'] = [d.month for d in df['date']]
    
    df.drop('date', axis=1, inplace=True)
    df = df.apply(zscore)
    df['stock'] = stock_name
    df = df.reset_index(drop=True)
    return df


def window_df(df):
    global all_inputs
    global all_labels
    for i in range(df.shape[0] - WINDOW_SIZE-1):
        rows = df.iloc[i:i+WINDOW_SIZE+1]
        input = rows[0:WINDOW_SIZE].to_numpy()
        input = np.reshape(input,(1,input.shape[0],input.shape[1]))
        label = rows[1:WINDOW_SIZE+1]["close"].to_numpy()
        if all_inputs is None:
            all_inputs = input
            all_labels = label
        else:
            all_inputs = np.vstack((all_inputs,input))
            all_labels = np.vstack((all_labels,label))
        

total_files = 0
discarded_files = 0
for root, _, files in os.walk(API_COMPANY_DATA_PATH):
    for file in tqdm.tqdm(files):
        with open(root+"\\"+file,'r') as f:
            data = json.load(f)
            total_files += 1
            if discard_for_api_error(data):
                discarded_files +=1
            else:
                df = make_data_frame(data, file[:-5]) # removes the .json extension from file name
                window_df(df)
                

print(all_inputs.shape)
print(all_labels.shape)
np.save(DATA_PATH+"stock_inputs.npy",all_inputs)
np.save(DATA_PATH+"stock_labels.npy",all_labels)


print("total files: ",total_files)
print("discarded files: ", discarded_files)
print("Well formed files: ",total_files-discarded_files)
            

