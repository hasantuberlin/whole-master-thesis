########################################
# Jharu: The Error Correction Using Wikipedia Revision History
# Md Kamrul Hasan
# kamrulhasancuetcse10@gmail.com
# June 2020-Present
# Master thesis, Big Data Management Group , TU Berlin
# Special thanks to:  Mohammad Mahdavi, moh.mahdavi.l@gmail.com, code repository :  https://github.com/BigDaMa/raha.git


########################################
# This module will extract table and infobox from wiki dump file. Then it will extract the revision data and extract the error and clean value by comparing two revision.
# This module then train edit distance, fasttext, elmo like wrod embedding method 
# We can retrained the pre trained model
# we can cross validate our testing and traning data set of wiki 
# we can correction a  dirty dataset in which error detected alreay have perfromed
# we can evaluate a model based on wiki model
# we can retrain on real world dirty dataset 
# we can  train model based on domain . for  example localtion, date etc. and apply on real world datasets
# This is the whole cleaning pipeline for our system
########################################
# library for fasttext
import warnings
warnings.filterwarnings('ignore')
#import fasttext
#from fasttext import train_unsupervised
import gensim
from gensim.models import FastText
import ast
########################################
#library for parsing and extracting wiki revision data
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
import datetime
from pandas import DataFrame
import random
import pickle
import pandas as pd
import logging

########################################
# This libraray is for edit distance model
from collections import Counter
from aion.util.spell_check import SpellCorrector
from fuzzywuzzy import fuzz
aion_dir = 'aion/'
sys.path.insert(0, aion_dir)
def add_aion(curr_path=None):
    if curr_path is None:
        dir_path = os.getcwd()
        target_path = os.path.dirname(os.path.dirname(dir_path))
        if target_path not in sys.path:
            #print('Added %s into sys.path.' % (target_path))
            sys.path.insert(0, target_path)
            
add_aion()
#############################################
# Library for cross validation
#scikit learn library
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, cross_validate, cross_val_score
#############################################
class Jharu:
    """
    The main class.
    """
    def __init__(self):
        """
        The constructor.
        """
        self.PRETRAINED_VALUE_BASED_MODELS_PATH = ""
        self.VALUE_ENCODINGS = ["identity", "unicode"]
        self.IGNORE_SIGN = "<<<IGNORE_THIS_VALUE>>>"
        self.VERBOSE = True
        self.SAVE_RESULTS = True
        self.ONLINE_PHASE = False
        self.REFINE_PREDICTIONS = True
        self.LABELING_BUDGET = 20
        self.MAX_VALUE_LENGTH = 50
        self.REVISION_WINDOW_SIZE = 5 
    def evaluate_model (self,model_type,total_error, total_error_to_repair, total_correction):
        if total_error_to_repair==0:
            precision=0
        else:
            precision=total_correction/total_error_to_repair
        if total_error==0:
            recall=0
        else:
            recall=total_correction/total_error
        if (precision+recall)==0:
            f_score=0
        else:
            f_score=(2 * precision * recall) / (precision + recall)      
        logfile_name=model_type+".log"
        logging.basicConfig(filename=logfile_name, level=logging.INFO)
        performance_string="Time: "+ str(datetime.datetime.now()) + " Model Type: " + str(model_type) +" Precision: " +str(precision) +" Recall: "+ str(recall) +"F-score: " +str(f_score)
        logging.info(performance_string)
        print("Performance: {}\nPrecision = {:.2f}\nRecall = {:.2f}\nF1 = {:.2f}".format(model_type,precision, recall, f_score))
    
    def split_train_test_data(self,json_file_list):
       random.shuffle(json_file_list)
       training = json_file_list[:int(len(json_file_list)*0.8)]
       testing = json_file_list[-int(len(json_file_list)*0.2):]
       return training, testing #tr,tt=spilt() 
    def error_correction_fasttext(self,model_type,datasets_type,dataset_1,dataset_2): #model=edit distance, fasttext, dataset_type=wiki_realword
        total_error=0
        total_error_to_repaired=0
        total_repaired=0
        try:
            if model_type=="Fasttext_All_Domain":
                if datasets_type=="wiki":
                    error_correction=self.prepare_testing_datasets_wiki(dataset_1,dataset_2) #dataset1 : json_list, dataset_1: path of json_filelist
                if datasets_type=="real_world":
                    error_correction=self.prepare_testing_datasets_real_world(dataset_1,dataset_2) #dataset_1 clean data for real world
                model_fasttext=FastText.load("model/Fasttext_All_Domain.w2v")
            if model_type=="Fasttext_Domain_Location":
                if datasets_type=="wiki":
                    error_correction=self.prepare_domain_testing_datasets_wiki(dataset_1,dataset_2,"location") #dataset1 : json_list, dataset_1: path of json_filelist
                    total_error=self.calculate_total_error_wiki(dataset_1,dataset_2)
                if datasets_type=="real_world":
                    error_correction=self.prepare_domain_testing_datasets_realworld(dataset_1,dataset_2) #dataset_1 clean data for real world
                    total_error=self.calculate_total_error_realworld(dataset_1,dataset_2)
                model_fasttext=FastText.load("model/Fasttext_Location_Domain.w2v")
            if model_type=="Fasttext_CV_Fold":
                model_fasttext=FastText.load("model/Fasttext_CV_Fold.w2v")
        except Exception as e:
            print('Model Error: ',str(e))
        for error_value, actual_value in zip(error_correction['error'],error_correction['actual']):
            if model_type=="Fasttext_Domain_Location":
                pass
            else:
                total_error=total_error+1
            try:
                if not any(x1.isdigit() for x1 in error_value):
                    total_error_to_repaired=total_error_to_repaired+1    
                    similar_value=model_fasttext.most_similar(error_value)
                    first,b=similar_value[0]
                    first=first.lower()
                    actual_value=actual_value.lower()
                    first=first.strip()
                    actual_value=actual_value.strip()
                    #print('Error : ', error_value, ' Fixed: ', first, ' Actual: ', actual_value)
                    if first==actual_value:
                        total_repaired=total_repaired+1
            except Exception as e:
                print('Error correction model: ',str(e))
                continue
        self.evaluate_model(model_type,total_error,total_error_to_repaired,total_repaired)
    def error_correction_fasttext_with_retrain_wiki(self,model_type,datasets_type,dataparam_1,dataparam_2):
        total_error=0
        total_error_to_repaired=0
        total_repaired=0
        if model_type=="Fasttext_All_Domain": #every time it will load the pretrained model to test new wiki table
            error_correction=self.prepare_testing_datasets_wiki(dataparam_1,dataparam_2) #dataparam_1 : json_list, dataparam_2: path of json_filelist
            model_fasttext=FastText.load("model/Fasttext_All_Domain.w2v")
        if model_type=="Fasttext_CV_Fold":
            model_fasttext=FastText.load("model/Fasttext_CV_Fold.w2v")
        if model_type=="Fasttext_Domain_Location":
            model_fasttext=FastText.load("model/Fasttext_Location_Domain.w2v")
            error_correction=self.prepare_domain_testing_datasets_wiki(dataparam_1,dataparam_2,"location")
            total_error=self.calculate_total_error_wiki(dataparam_1,dataparam_2)
        if datasets_type=="wiki":  
            train_data_rows=[]
            for rf in dataparam_1:               
                if rf.endswith(".json"):
                    try:
                        revision_list = json.load(io.open(os.path.join(dataparam_2, rf), encoding="utf-8"))    
                        one_item=revision_list[-1]
                        old_value=str(one_item[0]['old_value'].strip())
                        new_value=str(one_item[0]['new_value'].strip())
                        vicinity=one_item[0]['vicinity']
                        vicinity=remove_markup(str(vicinity))
                        vicinity= ast.literal_eval(vicinity)
                        #print('Before: ',vicinity)
                        train_vicinity_index = vicinity.index(old_value)
                        del vicinity[train_vicinity_index]
                        vicinity.append(new_value)
                        vicinity = [x for x in vicinity if not any(x1.isdigit() for x1 in x)]
                        vicinity=[x for x in vicinity if len(x)!=0] #remove empty item from list
                        #vicinity=[re.sub('[^a-zA-Z0-9.-]+', ' ', _) for _ in vicinity]
                        #print('After: ', vicinity)
                        #row=list(filter(None, row))
                        dirty_table=one_item[0]['dirty_table']
                        for index, row in enumerate(dirty_table):
                            if index==0:
                                continue
                            shape=len(row)
                            row=remove_markup(str(row))
                            row= ast.literal_eval(row)
                            row=list(filter(None, row))
                            #remove all digit 
                            row = [x for x in row if not any(x1.isdigit() for x1 in x)]
                            row=[x for x in row if len(x)!=0] #remove empty item from list
                            if row:
                                row=[re.sub('[^a-zA-Z0-9.-]+', ' ', _) for _ in row]
                                train_data_rows.append(row)
                    except Exception as e:
                        print('Exception: ',str(e))
            if train_data_rows:             
                model_fasttext.build_vocab(train_data_rows, update=True)
                model_fasttext.train(sentences=train_data_rows, total_examples = len(train_data_rows), epochs=5)
            for error_value, actual_value in zip(error_correction['error'],error_correction['actual']):
                try:
                    if model_type=="Fasttext_Domain_Location":
                        pass
                    else:
                        total_error=total_error+1
                    
                    if not any(x1.isdigit() for x1 in error_value):
                        total_error_to_repaired=total_error_to_repaired+1    
                        similar_value=model_fasttext.most_similar(error_value)
                        #print('Actual value: ',  actual_value,'Most similar value of : ',error_value, ' ' , similar_value)
                        first,b=similar_value[0]
                        #print('Error : ', error_value, ' Fixed: ', first, ' Actual: ', actual_value)
                        first=first.strip()
                        actual_value=actual_value.strip()
                        if first==actual_value:
                            print('Error : ', error_value, ' Fixed: ', first, ' Actual: ', actual_value)
                            total_repaired=total_repaired+1  
                except: 
                    continue                                     
        print(total_error,total_error_to_repaired, total_repaired )             
        model_type=model_type+' retrain wiki '
        self.evaluate_model(model_type,total_error,total_error_to_repaired,total_repaired)
    def error_correction_fasttext_with_retrain_realworld(self,datasets_type,dataset_1,dataset_2, model_type):
        total_error=0
        total_error_to_repaired=0
        total_repaired=0
        try:
            if model_type=="Fasttext_All_Domain":
                if datasets_type=="real_world":
                    error_correction=self.prepare_testing_datasets_real_world(dataset_1,dataset_2) #dataset_1 clean data for real world
                model_fasttext=FastText.load("model/Fasttext_All_Domain.w2v")
            if model_type=="Fasttext_Domain_Location":
                error_correction=self.prepare_domain_testing_datasets_realworld(dataset_1,dataset_2) #dataset_1 clean data for real world
                total_error=self.calculate_total_error_realworld(dataset_1,dataset_2)
                model_fasttext=FastText.load("model/Fasttext_Location_Domain.w2v")
        except Exception as e:
            print('Model Error: ',str(e))
        data_for_retrain=self.prepare_dataset_for_retrain_realworld(dataset_1,dataset_2)
        train_data_rows=[]
        try:         
            data_for_retrain=data_for_retrain.values.tolist()
            for row in data_for_retrain:
                row = list(map(str, row))
                row=list(filter(None, row))
                train_data_rows.append(row)
            if train_data_rows:
                if train_data_rows:
                    model_fasttext.build_vocab(train_data_rows, update=True)
                    model_fasttext.train(sentences=train_data_rows, total_examples = len(train_data_rows), epochs=5)
        except Exception as e:
            print("Exception from spell model : ", str(e))
        for error_value, actual_value in zip(error_correction['error'],error_correction['actual']):
            if model_type=="Fasttext_Domain_Location" and datasets_type=="real_world":
                pass
            else:
                total_error=total_error+1
            try:
                if not any(x1.isdigit() for x1 in error_value):
                    total_error_to_repaired=total_error_to_repaired+1    
                    similar_value=model_fasttext.most_similar(error_value)
                    first,b=similar_value[0]
                    first=first.lower()
                    actual_value=actual_value.lower()
                    #print('Error : ', error_value, ' Fixed: ', first, ' Actual: ', actual_value)
                    first=first.strip()
                    actual_value=actual_value.strip()
                    if first==actual_value:
                        print('Error : ', error_value, ' Fixed: ', first, ' Actual: ', actual_value)
                        total_repaired=total_repaired+1
            except Exception as e:
                print('Error correction model: ',str(e))
                continue
        self.evaluate_model(model_type,total_error,total_error_to_repaired,total_repaired)
    def prepare_testing_datasets_wiki(self, file_list_wiki,rd_folder_path):
        total_data=0
        actual_error = pd.DataFrame(columns = ['actual', 'error'])
        for rf in file_list_wiki:
            if rf.endswith(".json"):
                try:
                    revision_list = json.load(io.open(os.path.join(rd_folder_path, rf), encoding="utf-8"))    
                    for one_item in revision_list:
                        old_value=str(one_item[0]['old_value'].strip())
                        old_value=remove_markup(str(old_value))
                        old_value=re.sub('[^a-zA-Z0-9.-]+', ' ', old_value)
                        old_value=old_value.strip()

                        new_value=str(one_item[0]['new_value'].strip())
                        new_value=remove_markup(str(new_value))
                        new_value=re.sub('[^a-zA-Z0-9.-]+', ' ', new_value)
                        new_value=new_value.strip()
                        if  old_value and new_value and old_value !=" " and new_value!=" " and len(old_value)>3 and len(new_value)>3 and old_value!="none" and new_value!="none" and old_value!="None" and new_value!="None":
                            actual_error.loc[-1] = [new_value, old_value]
                            actual_error.index = actual_error.index + 1  # shifting index
                            actual_error = actual_error.sort_index()
                            total_data=total_data+1
                except Exception as e:
                    print('Exception from wiki: ', str(e))
        print("total_data: ",total_data)
        return actual_error
    def prepare_domain_testing_datasets_wiki(self, file_list_wiki,rd_folder_path,domain_type):
        total_data=0
        if domain_type=="location":
            domain_location=['Country', 'COUNTRY', 'country', 'CITY', 'City','city','Location','LOCATION','location','Place','PLACE','place','VENUE','venue','Venue','Town','town','TOWN', 'birth_place','death_place']
        actual_error = pd.DataFrame(columns = ['actual', 'error'])
        for rf in file_list_wiki:
            if rf.endswith(".json"):
                try:
                    revision_list = json.load(io.open(os.path.join(rd_folder_path, rf), encoding="utf-8"))    
                    for one_item in revision_list:
                        if domain_location:
                            if one_item[0]['errored_column'] in domain_location: 
                                old_value=str(one_item[0]['old_value'].strip())
                                old_value=remove_markup(str(old_value))
                                old_value=re.sub('[^a-zA-Z0-9.-]+', ' ', old_value)
                                old_value=old_value.strip()
                                new_value=str(one_item[0]['new_value'].strip())
                                new_value=remove_markup(str(new_value))
                                new_value=re.sub('[^a-zA-Z0-9.-]+', ' ', new_value)
                                new_value=new_value.strip()
                                if old_value and new_value and old_value !=" " and new_value!=" " and len(old_value)>3 and len(new_value)>3 and old_value!="none" and new_value!="none" and old_value!="None" and new_value!="None":
                                    actual_error.loc[-1] = [new_value, old_value]
                                    actual_error.index = actual_error.index + 1  # shifting index
                                    actual_error = actual_error.sort_index()
                                    total_data=total_data+1
                except Exception as e:
                    print('Exception from wiki: ', str(e))
        print("Total data to repair: ", total_data)
        return actual_error
    def prepare_domain_testing_datasets_realworld(self, clean_data_path, dirty_data_path):
        actual_error = pd.DataFrame(columns = ['actual', 'error'])
        clean_data=pd.read_csv(clean_data_path)
        dirty_data=pd.read_csv(dirty_data_path)
        #Hospital
        clean_data_col=['Address1','City','State','CountyName']
        dirty_data_col=['address_1','city','state','county']
        #clean_data_col=['city','state']
        #dirty_data_col=['city','state']
        for dir_col, clean_col in zip(dirty_data_col, clean_data_col):
            for indx, (value1, value2) in enumerate(zip(clean_data[clean_col],dirty_data[dir_col])):
                if value1==value2:
                    continue
                else:
                    actual_error.loc[-1] = [value1, value2]
                    actual_error.index = actual_error.index + 1  # shifting index
                    actual_error = actual_error.sort_index()
        return actual_error
    def calculate_total_error_realworld(self,clean_data_path, dirty_data_path):
        total_error=0
        clean_data=pd.read_csv(clean_data_path)
        dirty_data=pd.read_csv(dirty_data_path)
        clean_data_col=clean_data.columns.values
        dirty_data_col=dirty_data.columns.values
        for dir_col, clean_col in zip(dirty_data_col, clean_data_col):
            for indx, (value1, value2) in enumerate(zip(clean_data[clean_col],dirty_data[dir_col])):
                if value1==value2:
                    continue
                else:
                    total_error=total_error+1
                    
        return total_error
    def calculate_total_error_wiki(self,file_list_wiki,rd_folder_path):
        total_error=0
        for rf in file_list_wiki:
            if rf.endswith(".json"):
                try:
                    revision_list = json.load(io.open(os.path.join(rd_folder_path, rf), encoding="utf-8"))
                    for one_item in revision_list:
                        old_value=str(one_item[0]['old_value'].strip())
                        if old_value:
                            total_error=total_error+1                   
                except Exception as e:
                    print('Exception: ',str(e))  
        return total_error

    def prepare_testing_datasets_real_world(self,clean_data_path, dirty_data_path):
        actual_error = pd.DataFrame(columns = ['actual', 'error'])
        clean_data=pd.read_csv(clean_data_path)
        dirty_data=pd.read_csv(dirty_data_path)
        clean_data_col=clean_data.columns.values
        dirty_data_col=dirty_data.columns.values
        for dir_col, clean_col in zip(dirty_data_col, clean_data_col):
            for indx, (value1, value2) in enumerate(zip(clean_data[clean_col],dirty_data[dir_col])):
                if value1==value2:
                    continue
                else:
                    actual_error.loc[-1] = [value1, value2]
                    actual_error.index = actual_error.index + 1  # shifting index
                    actual_error = actual_error.sort_index()
        return actual_error
    def prepare_dataset_for_retrain_realworld(self, clean_data_path, dirty_data_path): #send two datasets
        clean_data=pd.read_csv(clean_data_path)
        dirty_data=pd.read_csv(dirty_data_path)
        if clean_data.shape != dirty_data.shape:
            sys.stderr.write("Two compared datasets do not have equal sizes!\n")
        else:
            clean_data_col=clean_data.columns.values
            dirty_data_col=dirty_data.columns.values
            for dir_col, clean_col in zip(dirty_data_col, clean_data_col):
                dirty_data.loc[dirty_data[dir_col] != clean_data[clean_col], dir_col] = None #replace error value with NaN
        return dirty_data
    def prepare_datasets_retrain_wiki(self,file_list_wiki,rd_folder_path): ###only for edit distance
        train_data_rows=[]
        for rf in file_list_wiki:
            if rf.endswith(".json"):
                try:
                    revision_list = json.load(io.open(os.path.join(rd_folder_path, rf), encoding="utf-8"))    
                    one_item=revision_list[-1]
                    #old_value=str(one_item[0]['old_value'].strip())
                    #new_value=str(one_item[0]['new_value'].strip())
                    dirty_table=one_item[0]['dirty_table']
                    for index, row in enumerate(dirty_table):
                        if index==0:
                            continue
                        row=remove_markup(str(row))
                        row= ast.literal_eval(row)
                        row=list(filter(None, row))
                        row = [x for x in row if not any(x1.isdigit() for x1 in x)]
                        if row:
                            row=[re.sub('[^a-zA-Z0-9.-]+', ' ', _) for _ in row]
                            train_data_rows.extend(row)
                except Exception as e:
                    print('Exception: ',str(e))  
        return train_data_rows


if __name__ == "__main__":
    app = Jharu()
    
    wiki_data_path_table="datasets/Table_for_creating_model/"
    filelist = os.listdir(os.path.join(wiki_data_path_table))
    #Split_train_test_datasets_wiki
    train_datasets,test_datasets=app.split_train_test_data(filelist)
    #################load tax data##################
    clean_data_path_tax="datasets/tax/clean.csv"
    dirty_data_path_tax="datasets/tax/dirty.csv"
    #################load hospital data#############
    clean_data_path_hos="datasets/hospital/clean.csv"
    dirty_data_path_hos="datasets/hospital/dirty.csv"
    ###########################################Edit Distance##########################################################
   
    ######################################################Fasttext#############################################
    #Create_model_Fasttext general
    #app.train_fasttext_all_domain(train_datasets,wiki_data_path_table)
    #testing fasttext model general_wiki data
    #print("Fasttext general wiki data")
    #app.error_correction_fasttext("Fasttext_All_Domain","wiki",test_datasets,wiki_data_path_table)
    #
    #testing hospital data general
    #print("Testing tax data fasttext domain")
    #app.error_correction_fasttext_with_retrain_realworld("real_world",clean_data_path_tax,dirty_data_path_tax, "Fasttext_Domain_Location")
    #app.error_correction_fasttext("Fasttext_Domain_Location","real_world",clean_data_path_hos,dirty_data_path_hos)
    #
    #Create model fasttext domain based_location
    #
    # app.train_domain_based_fasttext_model("location",train_datasets,wiki_data_path_table)
    #
    #testing fasttext model domain_wiki data
    #print("Fasttext domain location")
    #app.error_correction_fasttext("Fasttext_Domain_Location","wiki",test_datasets,wiki_data_path_table)
    #print("Testing Fasttext domain wiki retrain")
    #app.error_correction_fasttext_with_retrain_wiki("Fasttext_All_Domain","wiki",test_datasets,wiki_data_path_table)
    #testing fasttext model domain hospital  data 
    #app.error_correction_fasttext("Fasttext_Domain_Location","real_world",clean_data_path,dirty_data_path)
    #
    #retrain fasttext general with hospital data
    #print("Hospital")
    #app.error_correction_fasttext_with_retrain_realworld("real_world",clean_data_path_hos,dirty_data_path_hos, "Fasttext_Domain_Location")
    #
    #retrain fasttext domain with tax data
    #print("Tax data")
    #app.error_correction_fasttext_with_retrain_realworld("real_world",clean_data_path_tax,dirty_data_path_tax, "Fasttext_Domain_Location")
    #Create model elmo/Bert
    #
    #
    #
    # 
    #
    #
