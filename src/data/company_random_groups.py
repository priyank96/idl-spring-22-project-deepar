import os
import csv
import sys
import pickle
import random
import argparse
import numpy as np
import pandas as pd

from scipy.stats import zscore
from functools import reduce
from pathlib import Path

sys.path.insert(0,str(Path(__file__).parent.parent))
from constants import DATA_PATH, COMPANY_DATA_PATH, TRAIN_TEST_SPLIT

company_name_index = 0

index_to_company = dict()
company_to_index = dict()
sector_company_map = dict()

with open(os.path.join(DATA_PATH,'sectors.csv'),'r') as op:
    csv_reader = csv.reader(op) 
    for row in csv_reader:
        #  rows are like company, sector
        companies = sector_company_map.get(row[1],list())
        companies.append(row[0])
        sector_company_map[row[1]] = companies

def window_df(df, window_size, train= True):
    train_end_idx = int(TRAIN_TEST_SPLIT*df.shape[0])
    
    if train:
      for i in range(train_end_idx - window_size-1):
          input = df.iloc[i:i+window_size]
          label = df.iloc[i+1:i+window_size+1]
          yield input,label
          
    else:
      for i in range(train_end_idx,df.shape[0]-window_size-1):
          input = df.iloc[i:i+window_size]
          label = df.iloc[i+1:i+window_size+1]
          yield input,label


def _rename_columns(df:pd.DataFrame,i):
    i = str(i)
    rename_map = {
        'open' : i + '_open',
        # 'high' : i + '_high',
        # 'low' : i + '_low',
        # 'close' : i + '_close',
        # 'volume' : i + '_volume',
        'name' : i + '_name'
    }
    df.rename(columns=rename_map, inplace=True)
    return df

def _update_company_mappings(company_name):
    global company_name_index
    if company_name not in company_to_index.keys():
        company_to_index[company_name] = company_name_index
        index_to_company[company_name_index] = company_name
        company_name_index += 1

def make_data_frame(company_names, sector_name):
    dfs = []
    for i, company_name in enumerate(company_names):
        df = pd.read_csv(os.path.join(COMPANY_DATA_PATH,company_name+'.csv'))
        df = df[['date','open']]
        df["name"] = company_name
        
        
        # Add this company name to our mapping if needed
        _update_company_mappings(company_name)

        # Process date values
        df["date"] = pd.to_datetime(df["date"],infer_datetime_format=True)
        df = df.sort_values(by="date")
        #  Data transformations
        df[['open']] = df[['open']].apply(np.log)
        df[['open']] = df[['open']].apply(zscore)
        
        #  Rename columns so we can join
        df = _rename_columns(df,i)
        dfs.append(df)
    
    df_final = reduce(lambda left,right: pd.merge(left,right,on='date', how='inner'), dfs)
    df_final = df_final.sort_values(by="date")
    #  Process date values
    df_final['day'] = [d.day for d in df_final['date']]
    df_final['month'] = [d.month for d in df_final['date']]
    df_final = df_final.drop('date',axis=1)
    
    
    # add sector column
    df_final['sector'] = sector_name
    df.reset_index(drop=True, inplace=True)
    return df_final



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-gs", "--group_size", help="Number of companies per group")
    # Num rows is suggestion. It guarantees min number of rows , not max
    parser.add_argument("-r","--num_rows",help="Number of training rows")
    parser.add_argument("-w","--window_size",help="Number of time steps in a window")
    args = parser.parse_args()

    group_size = int(args.group_size)
    num_rows = int(args.num_rows)
    window_size = int(args.window_size)

    total_rows = 0
    all_inputs = []
    all_labels = []
    all_test_inputs = []
    all_test_labels = []

    while total_rows < num_rows:
        print('total rows: ',total_rows)
        keys = list(sector_company_map.keys())
        sector = random.choice(keys)
        
        # This could run infinitely
        while len(sector_company_map[sector]) < group_size:
            sector = random.choice(keys)
        
        companies = random.sample(population=sector_company_map[sector],k=group_size)
        #  make the dataframe
        df = make_data_frame(companies,sector)
        #  see if it has at least window size rows
        if len(df) < window_size:
          continue
        if df.isna().values.any():
          print("found nans")
          continue
        # Window this df and add to test and train npy arrays
        for input, label in window_df(df,window_size,True):
            all_inputs.append(input)
            all_labels.append(label)
            total_rows += 1
        for input, label in window_df(df,window_size,False):
            all_test_inputs.append(input)
            all_test_labels.append(label)
    print(df.info())
    # Start saving things
    print(index_to_company)
    with open(DATA_PATH+"/index_to_company.pkl", "wb") as f:
        pickle.dump(index_to_company, f)

    print(company_to_index)
    with open(DATA_PATH+"/company_to_index.pkl", "wb") as f:
        pickle.dump(company_to_index, f)
    
    all_labels = np.array(all_labels)
    print("train labels: ",all_labels.shape)
    np.save(DATA_PATH+"/stock_labels.npy",all_labels)
    del all_labels
                    
    all_inputs = np.array(all_inputs)
    print("train inputs: ",all_inputs.shape)
    np.save(DATA_PATH+"/stock_inputs.npy",all_inputs)
    del all_inputs

    all_test_labels = np.array(all_test_labels)
    print("test labels: ",all_test_labels.shape)
    np.save(DATA_PATH+"/stock_test_labels.npy",all_test_labels)
    del all_test_labels
                    
    all_test_inputs = np.array(all_test_inputs)
    print("test inputs: ",all_test_inputs.shape)
    np.save(DATA_PATH+"/stock_test_inputs.npy",all_test_inputs)
    del all_test_inputs