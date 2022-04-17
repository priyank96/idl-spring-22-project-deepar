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
    3. Apply log to each column
    4. Iterate over the rows of this dataframe with a window size of n, moving down one row at a time.
    5. For each window make two numpy matrices of shape (n-1 x ):
        1. Input: row 0 to n-1 with [month,day,open,volume]
        2. Output: row 1 to n [open,close]
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
import gc
from functools import reduce

sys.path.insert(0,str(Path(__file__).parent.parent))
from constants import API_COMPANY_DATA_PATH, WINDOW_SIZE, DATA_PATH, TRAIN_TEST_SPLIT, TARGET_COMPANIES

def discard_for_api_error(data):
    if "Information" in data:
        return True
    if "Error Message" in data:
        return True
    return False

def discard_for_df_error(df,stock_index):
  return df[str(stock_index)+"_volume"].isnull().any() or df[str(stock_index)+"_open"].isnull().any()

def make_data_frame(data, stock_index):
    rows = []
    for day_data_key in data["Time Series (Daily)"]:
        day_data = data["Time Series (Daily)"][day_data_key]

        rows.append([
            day_data_key,
            float(day_data["1. open"]),
            # adding 1 here because when apply a log transofrm we get a -inf, adding 1 makes that 0
            float(day_data["5. volume"])+1
        ])
    
    df = pd.DataFrame(rows, columns=["date",str(stock_index)+"_open",str(stock_index)+"_volume"])
    del rows
    df["date"] = pd.to_datetime(df["date"],infer_datetime_format=True)
    df = df.sort_values(by="date")
    df['day'] = [d.day for d in df['date']]
    df['month'] = [d.month for d in df['date']]
    
    return df


def window_df(df, train= True, num_companies=5):
    train_end_idx = int(TRAIN_TEST_SPLIT*df.shape[0])
    column_names = []
    for i in range(num_companies):
      column_names.append(str(i)+"_open")
      column_names.append(str(i)+"_volume")
    if train:
      for i in range(train_end_idx - WINDOW_SIZE-1):
          rows = df.iloc[i:i+WINDOW_SIZE+1]
          input = rows[0:WINDOW_SIZE]
          label = rows[1:WINDOW_SIZE+1][column_names]
          yield input,label
          
    else:
      for i in range(train_end_idx,df.shape[0]-WINDOW_SIZE-1):
          rows = df.iloc[i:i+WINDOW_SIZE+1]
          input = rows[0:WINDOW_SIZE]
          label = rows[1:WINDOW_SIZE+1][column_names]
          yield input,label

if __name__ =='__main__':
  num_companies = 5
  all_inputs = []
  all_labels = []
  all_test_inputs = []
  all_test_labels = []

  total_files = 0
  discarded_files = 0
  index_to_company = dict() 
  company_to_index = dict()

  company_name_index = 0
  i = 0

  root = None
  files = None
  for r, _, f in os.walk(API_COMPANY_DATA_PATH):
    root = r
    files = f

  if TARGET_COMPANIES:
    files = [file + '.json' for file in TARGET_COMPANIES]

  
  dfs = []
  for file in tqdm.tqdm(files):
      with open(root+"/"+file,'r') as f:

          data = json.load(f)
          total_files += 1
          if discard_for_api_error(data):
              discarded_files +=1
          else:
              index_to_company[company_name_index] = file[:-5] # removes the .json extension from file name
              company_to_index[file[:-5]] = company_name_index
              df = make_data_frame(data, company_name_index) 
              if discard_for_df_error(df,company_name_index):
                # Remove mapping, so we don't access this company in the test set
                del company_to_index[file[:-5]]
                discarded_files +=1
                continue
              company_name_index += 1
              # merge columns of this new df with the final_df
              dfs.append(df)

  column_names = ['month','day']
  for i in range(num_companies):
    column_names.append(str(i)+"_open")
    column_names.append(str(i)+"_volume")

  df_final = reduce(lambda left,right: pd.merge(left,right,on=['date','day','month']), dfs)
  df_final = df_final.sort_values(by="date")
  df_final = df_final.drop('date',axis=1)
  df_final = df_final[column_names]

  df_final = df_final.apply(np.log)
  df_final = df_final.apply(zscore)
  for input, label in window_df(df_final):
    all_inputs.append(input)
    all_labels.append(label)


  print(index_to_company)
  with open(DATA_PATH+"/index_to_company.pkl", "wb") as f:
      pickle.dump(index_to_company, f)

  print(company_to_index)
  with open(DATA_PATH+"/company_to_index.pkl", "wb") as f:
      pickle.dump(company_to_index, f)

  gc.collect()

  all_labels = np.array(all_labels)
  print("train labels: ",all_labels.shape)
  np.save(DATA_PATH+"/stock_labels.npy",all_labels)
  del all_labels
  gc.collect()
                  
  all_inputs = np.array(all_inputs)
  print("train inputs: ",all_inputs.shape)
  np.save(DATA_PATH+"/stock_inputs.npy",all_inputs)
  del all_inputs
  gc.collect()

  for input, label in window_df(df_final,False):
    all_test_inputs.append(input)
    all_test_labels.append(label)
  
  all_test_labels = np.array(all_test_labels)
  print("test labels: ",all_test_labels.shape)
  np.save(DATA_PATH+"/stock_test_labels.npy",all_test_labels)
  del all_test_labels
  gc.collect()
                  
  all_test_inputs = np.array(all_test_inputs)
  print("test inputs: ",all_test_inputs.shape)
  np.save(DATA_PATH+"/stock_test_inputs.npy",all_test_inputs)
  del all_test_inputs
  gc.collect()

  print("total files: ",total_files)
  print("discarded files: ", discarded_files)
  print("Well formed files: ",total_files-discarded_files)
       
