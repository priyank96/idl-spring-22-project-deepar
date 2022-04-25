import os
import yfinance as yf
import tqdm
import os
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).parent.parent))
from constants import DATA_PATH, COMPANY_DATA_PATH

DATA_PATH = 'C:\\Users\\priya\\Documents\\IDL\\Project\\data\\'
COMPANY_DATA_PATH = 'C:\\Users\\priya\\Documents\\IDL\\Project\\data\\company_data'
files = None
for r, _, f in os.walk(COMPANY_DATA_PATH):
  root = r
  files = f

with open(DATA_PATH+'sectors.csv','w') as op:
    for f in tqdm.tqdm(files):
        try:
            company = f[:-4]
            tickerdata = yf.Ticker(company) #the tickersymbol for Tesla
            op.write(company+","+tickerdata.info['sector']+"\n")
        except Exception as e:
            print(e)