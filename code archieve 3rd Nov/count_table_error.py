import os
revision_data_folder="datasets/revision-data/table"
page_count=0
table_count=0
table_with_error_count=0
max_r=0
total_total_with_revision=0
for folder in os.listdir(revision_data_folder):
    page_count=page_count+1
    page_folder=os.path.join(revision_data_folder,folder)
    if os.path.isdir(os.path.join(revision_data_folder, folder)):
        for nested_folder in os.listdir(os.path.join(revision_data_folder,folder)):
            table_count=table_count+1
            if os.path.isdir(os.path.join(page_folder, nested_folder)):
                filelist = os.listdir(os.path.join(page_folder, nested_folder))
                total_total_with_revision=len(filelist)
                if total_total_with_revision>max_r:
                    max_r=total_total_with_revision

                print(total_total_with_revision)
                table_with_error_count=table_with_error_count+total_total_with_revision
                total_total_with_revision=0
print('Page: ',page_count,' Table: ', table_count,' table_with_error_count', table_with_error_count,' Max ', max_r)
        


