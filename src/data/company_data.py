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
import pickle

sys.path.insert(0,str(Path(__file__).parent.parent))
from constants import API_COMPANY_DATA_PATH, WINDOW_SIZE, DATA_PATH, TRAIN_TEST_SPLIT

all_inputs = []
all_labels = []

all_test_inputs = []
all_test_labels = []

def discard_for_api_error(data):
    if "Information" in data:
        return True
    if "Error Message" in data:
        return True
    return False

def discard_for_df_error(df):
  return df["volume"].isnull().any() or df["close"].isnull().any()

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
    df['stock'] = stock_name # -1th index of covariates is the symbol of the stock
    df = df.reset_index(drop=True)
    return df


def window_df(df):
    global all_inputs
    global all_labels
    global all_test_inputs
    global all_test_labels

    train_end_idx = int(TRAIN_TEST_SPLIT*df.shape[0])
    for i in range(train_end_idx - WINDOW_SIZE-1):
        rows = df.iloc[i:i+WINDOW_SIZE+1]
        input = rows[0:WINDOW_SIZE].to_numpy()
        label = rows[1:WINDOW_SIZE+1]["open"].to_numpy()
        all_inputs.append(input)
        all_labels.append(label)
    
    for i in range(train_end_idx,df.shape[0]-WINDOW_SIZE-1):
        rows = df.iloc[i:i+WINDOW_SIZE+1]
        input = rows[0:WINDOW_SIZE].to_numpy()
        label = rows[1:WINDOW_SIZE+1]["open"].to_numpy()
        all_test_inputs.append(input)
        all_test_labels.append(label)

total_files = 0
discarded_files = 0
company_names= set()
i = 0
for root, _, files in os.walk(API_COMPANY_DATA_PATH):
    for file in tqdm.tqdm(files):
        with open(root+"/"+file,'r') as f:
            data = json.load(f)
            total_files += 1
            if discard_for_api_error(data):
                discarded_files +=1
            else:
                df = make_data_frame(data, file[:-5]) # removes the .json extension from file name
                if discard_for_df_error(df):
                  discarded_files +=1
                  continue
                window_df(df)
                company_names.add(file[:-5])


company_names = list(company_names)
company_index = {}
for i in range(len(company_names)):
  company_index[company_names[i]] = i

print(company_index)

with open(DATA_PATH+"/company_names.pkl", "wb") as f:
    pickle.dump(company_index, f)
           






                
                
all_inputs = np.array(all_inputs)
all_labels = np.array(all_labels)
all_test_inputs = np.array(all_test_inputs)
all_test_labels = np.array(all_test_labels)

print("train inputs: ",all_inputs.shape)
print("train labels: ",all_labels.shape)
print("test inputs: ",all_test_inputs.shape)
print("test labels: ",all_test_labels.shape)

np.save(DATA_PATH+"/stock_inputs.npy",all_inputs)
np.save(DATA_PATH+"/stock_labels.npy",all_labels)

np.save(DATA_PATH+"/stock_test_inputs.npy",all_test_inputs)
np.save(DATA_PATH+"/stock_test_labels.npy",all_test_labels)


print("total files: ",total_files)
print("discarded files: ", discarded_files)
print("Well formed files: ",total_files-discarded_files)
            

