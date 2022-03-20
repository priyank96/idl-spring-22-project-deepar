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
'''
from pathlib import Path
import sys
import os
import json 
import tqdm
import pandas as pd

sys.path.insert(0,str(Path(__file__).parent.parent))
from constants import API_COMPANY_DATA_PATH, COMPANY_DATA_PATH


def discard_for_api_error(data):
    if "Information" in data:
        return True
    if "Error Message" in data:
        return True
    return False

def make_data_frame(data, file_name):
    rows = []
    for day_data_key in data["Time Series (Daily)"]:
        day_data = data["Time Series (Daily)"][day_data_key]
        rows.append([
            day_data_key,
            day_data["1. open"], 
            day_data["2. high"],
            day_data["3. low"],
            day_data["4. close"],
            day_data["5. volume"]
        ])
    df = pd.DataFrame(rows, columns=["date","open","high","low","close","volume"])
    df["date"] = pd.to_datetime(df["date"],infer_datetime_format=True)
    df =df.sort_values(by="date")

    df.to_csv(COMPANY_DATA_PATH+"\\"+file_name+".csv", index=False)

        

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
                make_data_frame(data, file[:-5]) # removes the .json extension from file name


print("total files: ",total_files)
print("discarded files: ", discarded_files)
print("Well formed files: ",total_files-discarded_files)
            

