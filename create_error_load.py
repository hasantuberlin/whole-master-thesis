import pandas as pd
from pandas import DataFrame

convert_csv = pd.DataFrame(columns = ['_tid_','attribute'])
clean_data=pd.read_csv("datasets/tax/clean.csv")
dirty_data=pd.read_csv("datasets/tax/dirty.csv")


clean_data_col=clean_data.columns.values
dirty_data_col=dirty_data.columns.values
for dir_col, clean_col in zip(dirty_data_col, clean_data_col):
    for indx, (value1, value2) in enumerate(zip(clean_data[clean_col],dirty_data[dir_col])):
        value1=str(value1)
        value2=str(value2)
        if value1==value2:
            continue
        else:
            convert_csv.loc[-1] = [indx, clean_col]
            convert_csv.index = convert_csv.index + 1  # shifting index
            convert_csv = convert_csv.sort_index()
            
convert_csv.to_csv("tax_error_load.csv", sep=",", header=True, index=False, encoding="utf-8")