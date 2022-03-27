
'''
The goal of visualization is to understand the distribution of the different variables of our data:
Open, Close, High, Low, Volume.
This will inform the model architecture i.e: 
Will the model learn to predict Gaussian distributions or Poisson distributions?

'''
import numpy as np

from pathlib import Path
import sys
import plotly.express as px

sys.path.insert(0,str(Path(__file__).parent.parent))
from constants import DATA_PATH

test_data = np.load(DATA_PATH + '\\stock_inputs.npy', allow_pickle=True)
print(test_data.shape)

# figure for opening price
fig = px.histogram(x=test_data[:,:,2])
fig.show()

