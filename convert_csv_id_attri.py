import pandas as pd
from pandas import DataFrame

convert_csv = pd.DataFrame(columns = ['tid','attribute','correct_val'])
data=pd.read_csv("datasets/tax/clean.csv")
data_columns=data.columns.tolist()
data_list=data.values.tolist()
for row in data_list:
  index=0
  for col, item in zip(data_columns,row):
    tid=index
    attribute=col 
    current_value=item
    convert_csv.loc[-1] = [tid, attribute,current_value]
    convert_csv.index = convert_csv.index + 1  # shifting index
    convert_csv = convert_csv.sort_index()
    #print(convert_csv)

convert_csv.to_csv("tax_clean.csv", sep=",", header=True, index=False, encoding="utf-8")
