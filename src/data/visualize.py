
'''
The goal of visualization is to understand the distribution of the different variables of our data:
Open, Close, High, Low, Volume.
This will inform the model architecture i.e: 
Will the model learn to predict Gaussian distributions or Poisson distributions?
The covariate variables are arranged as: 
[month,day,open,low,high,close,volume, company name]
'''
import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import plotly.express as px

sys.path.insert(0,str(Path(__file__).parent.parent))
from constants import DATA_PATH

def save_plot(data, covariate_column_index, out_file):
  column = np.reshape(test_data[:,:,covariate_column_index],(-1,1))
  df = pd.DataFrame(data=column)
  fig = px.histogram(x=df[0],range_x=[df.min(),df.max()])
  fig.write_image(DATA_PATH+"/"+out_file)


if __name__ == '__main__':
  test_data = np.load(DATA_PATH + '/stock_inputs.npy', allow_pickle=True)
  print(test_data.shape)
  '''
  The covariate variables are arranged as: 
  [month,day,open,low,high,close, volume company name]
  '''
  indices_and_file_names = [(2,"open.png"),(3,"low.png"),(4,"high.png"),(5,"close.png"),(6,"volume.png")]
  for index, file_name in tqdm.tqdm(indices_and_file_names):
    save_plot(test_data,index,file_name)
