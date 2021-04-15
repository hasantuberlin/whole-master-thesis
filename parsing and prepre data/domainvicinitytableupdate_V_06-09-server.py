########################################
import os
import re
import io
import sys
import math
import json
import html
import pickle
import difflib
import unicodedata
import bs4
import bz2
import py7zr
import numpy
import mwparserfromhell
import libarchive.public
#import dataset
import wikitextparser as wtp
from pprint import pprint
from wikitextparser import remove_markup, parse
from pandas import DataFrame
import html
#from simplediff import diff as simplediff
#from difflib_data import *
########################################
########################################
class WikiErrorCorrection:
    """
    The main class.
    """
    def __init__(self):
        """
        The constructor.
        """
        #self.PRETRAINED_VALUE_BASED_MODELS_PATH = ""
        #self.VALUE_ENCODINGS = ["identity", "unicode"]
        #self.IGNORE_SIGN = "<<<IGNORE_THIS_VALUE>>>"
        self.VERBOSE = True
        #self.SAVE_RESULTS = True
        #self.ONLINE_PHASE = False
        #self.REFINE_PREDICTIONS = True
        #self.LABELING_BUDGET = 20
        #self.MAX_VALUE_LENGTH = 50
        #self.REVISION_WINDOW_SIZE = 5

    def make_list_from_sorted_json(self, revision_data_folder):
        table_folder_count=0
        self.table_count_with_error=0
        table_count_with_revision=0
        rd_folder_path = revision_data_folder
        modified_data_path=os.path.join('datasets','Table_04_09')
        if not os.path.isdir(modified_data_path):
            os.mkdir(modified_data_path)
        for folder in os.listdir(rd_folder_path): #initial archieve folder
            if '04Sep' in folder:
                print(folder)
                rdd_folder_path=os.path.join(rd_folder_path,folder)
                if os.path.isdir(os.path.join(rd_folder_path, folder)): #datasets/revision-data/archieve-foldername
                    for nested_folder in os.listdir(os.path.join(rd_folder_path,folder)):
                        print(nested_folder)
                        revision_list=[]
                        #print(nested_folder)
                        page_folder=os.path.join(rdd_folder_path,nested_folder) #datasets/revision-data/archieve-foldername/page_folder
                        #print(page_folder)
                        if os.path.isdir(os.path.join(rdd_folder_path, nested_folder)):
                            for nested_nested_folder in os.listdir(os.path.join(rdd_folder_path,nested_folder)):
                                table_folder_count=table_folder_count+1
                                if os.path.isdir(os.path.join(page_folder, nested_nested_folder)):
                                    ##########################
                                      #Check the parent id and revision id
                                    ##########################
                                    filelist = os.listdir(os.path.join(page_folder, nested_nested_folder))
                                    table_count_with_revision=table_count_with_revision+ len(filelist)
                                    filelist = sorted(filelist,key=lambda x: int(os.path.splitext(x)[0].split('_')[1]))#sort the json file list with revision id
                                    self.previous_revision_file=None
                                    self.previous_json_name=None
                                    for rf in filelist:
                                        self.current_json_name=rf
                                        current_parent_id=rf.split('.')[0].split('_')[0]
                                        #print('current parent id', current_parent_id)
                                        
                                        #current_revision_id=rf.split('.')[0].split('_')[1]
                                        #print(rf, 'parent_id:', current_parent_id, 'revision_id: ', current_revision_id)
                                        if rf.endswith(".json"):                                      
                                            self.current_revision_file=json.load(io.open(os.path.join(page_folder, nested_nested_folder, rf), encoding="utf-8"))
                                            if self.previous_revision_file and self.previous_json_name:
                                                #previous_parent_id= self.previous_json_name.split('.')[0].split('_')[0]
                                                previous_revision_id= self.previous_json_name.split('.')[0].split('_')[1]
                                                #print('previous_revision_id', previous_revision_id)
                                                if previous_revision_id == current_parent_id:
                                                    revision_list_from_fun=[]
                                                    revision_list_from_fun=self.diff_check_revision()
                                                    if revision_list_from_fun:
                                                        revision_list.append(revision_list_from_fun)
                                                    self.previous_revision_file=self.current_revision_file
                                                    self.previous_json_name=self.current_json_name
                                            else:
                                                self.previous_revision_file= self.current_revision_file
                                                self.previous_json_name=self.current_json_name
                                    
                        if revision_list:
                            json.dump(revision_list, open(os.path.join(modified_data_path, nested_folder + ".json"), "w", encoding='utf8'))
                                           
        print('Total table: ', table_folder_count, 'With revision: ', table_count_with_revision, 'Error Table: ', self.table_count_with_error)
        txt_file="table_folder_count: "+ str(table_folder_count)+ ". With Revision: "+ str(table_count_with_revision) + "Error Table: " + str(self.table_count_with_error)
        with open("table_count_04_09.txt", "w") as text_file:
            text_file.write(txt_file)
    def diff_check_revision(self):
        create_revision_list=[]
        table_column_current=None
        table_column_previous=None
        code_current =mwparserfromhell.parse(self.current_revision_file[0], skip_style_tags=True)
        code_previous=mwparserfromhell.parse(self.previous_revision_file[0], skip_style_tags=True)
        try:
            ########### Current revision table  data extraction
            table1=code_current.filter_tags(matches=lambda node: node.tag=="table")
            table_code_current = wtp.parse(str(table1[0])).tables[0]
            table_data_current=table_code_current.data()
            table_column_current=table_data_current[0]
            ########## previous revision table data extraction
            table2=code_previous.filter_tags(matches=lambda node: node.tag=="table")
            table_code_previous = wtp.parse(str(table2[0])).tables[0]
            table_data_previous=table_code_previous.data()
            table_column_previous=table_data_previous[0]
            df_data=DataFrame(table_data_previous)
            header=df_data.iloc[0]
            new_column_list=header.tolist()
            df_data=df_data[1:]
            df_data.columns=header
        except Exception as e:
            print('Exception from table data: ', str(e))
        if table_column_current and table_column_previous and len(table_column_previous) == len(set(table_column_previous)):
            self.table_count_with_error=self.table_count_with_error+1
            if len(table_column_current)==len(table_column_previous):
                text1=table_data_previous
                text2=table_data_current 
            
                if text1 and text2:
                    for index1, (txt1, txt2) in enumerate(zip(text1,text2)): #row parsing
                        if index1==0:
                            continue
                        d = difflib.Differ()
                        for index, (cell1,cell2) in enumerate(zip(txt1,txt2)): # values of row parsing
                            create_revision_dict={}
                            old_value=None
                            new_value=None
                            cell1=remove_markup(str(cell1))
                            cell2=remove_markup(str(cell2))
                            #print(cell1)
                            #print(cell2)
                            if cell1 and cell2 :
                                diff1 = d.compare([''.join(cell1)], [cell2])
                                try:
                                    if diff1:
                                        for line in diff1:
                                            #print(line)
                                            #print('###############################################################################')
                                            if not line.startswith(' '):
                                                if line.startswith('-'):
                                                    old_value=line[1:]
                                                if line.startswith('+'):
                                                    new_value=line[1:]
                                        if old_value and new_value:
                                            #table_column_current1=remove_markup(str(table_column_current))
                                            txt1=remove_markup(str(txt1))
                                            old_value=remove_markup(str(old_value))
                                            new_value=remove_markup(str(new_value))
                                            column_name=new_column_list[index]
                                            column_name=str(column_name)
                                            #print(column_name)
                                            #print(type(column_name))

                                            column_values=df_data[column_name].tolist()
                                            column_values=remove_markup(str(column_values))
                                            #value = html.unescape(value)
                                            #new_value = re.sub("[\t\n ]+", " ", new_value, re.UNICODE)
                                            #value = value.strip("\t\n ")
                                            cleanr = re.compile('<.*?>')
                                            
                                            all_column=list(df_data.columns)
                                            #all_column=html.unescape(str(all_column))
                                            #all_column=remove_markup(str(all_column))
                                            all_column = re.sub(cleanr, ' ', str(all_column))
                                            all_column=remove_markup(all_column)
                                            column_name=re.sub(cleanr, ' ', str(column_name))
                                            column_name=remove_markup(column_name)
                                            if len(old_value)<50 and len(new_value)<50:
                                                create_revision_dict={ "columns": all_column, "domain": column_values, "vicinity": txt1, "errored_column": column_name,"old_value": old_value, "new_value": new_value}
                                                create_revision_list.append(create_revision_dict)
                                                print('column: ',column_name,'old cell: ',old_value,  'new_cell: ', new_value)
                                except Exception as e:
                                    print('Exception from revised value: ', str(e))                 
        return create_revision_list
if __name__ == "__main__":
    app = WikiErrorCorrection()
    #app.extract_revisions(wikipedia_dumps_folder="datasets")
    app.make_list_from_sorted_json(revision_data_folder="datasets/revision-data/04-09-2020")
