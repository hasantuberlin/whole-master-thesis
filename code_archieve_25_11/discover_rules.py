from efficient_apriori import apriori

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


transactions_from_df = [tuple(row) for row in dirty_data.values.tolist()]


itemsets, rules = apriori(transactions_from_df, min_support=0.2, min_confidence=1)
print(rules)