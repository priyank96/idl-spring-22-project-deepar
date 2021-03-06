# IDL Spring 2022 Project - Probabilistic Forecasting of stock prices with Deep AR.

## Get started
1. Modify Parent Folder path in src/constants.py to current directory.

## Scripts for data preparation
1. [company_data.py](https://github.com/priyank96/idl-spring-22-project-deepar/blob/main/src/data/company_data.py) - To process data and generate windowed time series. 
2. [company_random_groups.py](https://github.com/priyank96/idl-spring-22-project-deepar/blob/main/src/data/company_random_groups.py) - To process data and generate windowed time series for grouped vector inputs.

Note: The appropriate data generation scripts are executed by the appropriate notebooks below

## Link to respective notebooks
1. [DeepAR Baseline](https://github.com/priyank96/idl-spring-22-project-deepar/blob/main/python_notebooks/Single_Company/DL_192Win_DeepAR_Baseline.ipynb) 
2. [Higher Autoregressive Context](https://github.com/priyank96/idl-spring-22-project-deepar/blob/main/python_notebooks/Single_Company/DL_192WinLen_Context.ipynb) 
3. [Deeper Horizon Weighted Loss](https://github.com/priyank96/idl-spring-22-project-deepar/blob/main/python_notebooks/Single_Company/DL_192WinLen_WeightLoss.ipynb) 
4. [Grouped Vector Input](https://github.com/priyank96/idl-spring-22-project-deepar/blob/main/python_notebooks/Vector_Group_Inputs/DL_192Vectors_Group_Data_SectorEmbed_Normal_Loss.ipynb)

Note: There are other notebooks for experiments that we have performed that we have not included in our paper.
