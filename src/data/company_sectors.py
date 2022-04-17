import os
import yfinance as yf
import tqdm
import os
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).parent.parent))
from constants import DATA_PATH, API_COMPANY_DATA_PATH

files = None
for r, _, f in os.walk(API_COMPANY_DATA_PATH):
  root = r
  files = f

with open(DATA_PATH+'sectors.csv','w') as op:
    for f in tqdm.tqdm(files):
        try:
            company = f[:-5]
            tickerdata = yf.Ticker(company) #the tickersymbol for Tesla
            op.write(company+","+tickerdata.info['sector']+"\n")
        except Exception as e:
            print(e)