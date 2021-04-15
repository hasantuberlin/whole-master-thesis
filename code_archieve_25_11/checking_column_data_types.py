import pandas as pd
#################load tax data##################
clean_data_path_tax="datasets/tax/clean.csv"
dirty_data_path_tax="datasets/tax/dirty.csv"
################load hospital data#############
clean_data_path_hos="datasets/hospital/clean.csv"
dirty_data_path_hos="datasets/hospital/dirty.csv"
actual_error = pd.DataFrame(columns = ['actual', 'error','area_code','city','state','zip','index'])
clean_data=pd.read_csv(clean_data_path_tax)
dirty_data=pd.read_csv(dirty_data_path_hos)
#Hospital
#clean_data_col=['City','State','ZipCode']
#dirty_data_col=['city','state','zip']
clean_data_col=['city','state']
dirty_data_col=['city','state']
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

#dirty_data.astype(str).astype(int)
print(is_numeric_dtype(dirty_data['zip']))

#for dir_col, clean_col in zip(dirty_data_col, clean_data_col):
  # pass