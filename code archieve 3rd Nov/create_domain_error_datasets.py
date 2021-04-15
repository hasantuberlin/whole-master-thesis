import pandas as pd
#################load tax data##################
clean_data_path_tax="datasets/tax/clean.csv"
dirty_data_path_tax="datasets/tax/dirty.csv"
################load hospital data#############
clean_data_path_hos="datasets/hospital/clean.csv"
dirty_data_path_hos="datasets/hospital/dirty.csv"
actual_error = pd.DataFrame(columns = ['actual', 'error','area_code','city','state','zip','index'])
clean_data=pd.read_csv(clean_data_path_tax)
dirty_data=pd.read_csv(dirty_data_path_tax)
#Hospital
#clean_data_col=['City','State','ZipCode']
#dirty_data_col=['city','state','zip']
clean_data_col=['city','state']
dirty_data_col=['city','state']
for dir_col, clean_col in zip(dirty_data_col, clean_data_col):
    for indx, (value1, value2) in enumerate(zip(clean_data[clean_col],dirty_data[dir_col])):
        value1=str(value1)
        value2=str(value2)
        if value1==value2:
            continue
        else:
            area_code=dirty_data.at[indx,'area_code']
            city=dirty_data.at[indx,'city']
            state=dirty_data.at[indx,'state']
            zip_1=dirty_data.at[indx,'zip']
            actual_error.loc[-1] = [value1, value2,area_code,city,state,zip_1, indx]
            actual_error.index = actual_error.index + 1  # shifting index
            actual_error = actual_error.sort_index()
actual_error.to_csv("tax_domain_error_28_10.csv")