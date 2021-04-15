##################
# from my model
###########################
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
import ast
##################
import warnings
warnings.filterwarnings('ignore')
import fasttext
from fasttext import train_unsupervised
import gensim
from gensim.models import FastText
################################
import pandas as pd
import numpy as np
import os
import calendar
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import re
from collections import Counter
import logging
import random
#from happytransformer import HappyBERT
#from happytransformer import HappyROBERTA
##########################################################


class BertCorrection:
    def __init___(self):
        self.TAG_RE = re.compile(r'<[^>]+>')
        
        
        #self.bert_base_cased = HappyBERT("bert-base-cased")
        #self.bert_large_uncased = HappyBERT("bert-large-uncased")
        #self.bert_large_cased = HappyBERT("bert-large-cased")
    def split_train_test_data(self,json_file_list):
       random.shuffle(json_file_list)
       training = json_file_list[:int(len(json_file_list)*0.9)]
       testing = json_file_list[-int(len(json_file_list)*0.1):]
       return training, testing #tr,tt=spilt()
    def remove_tags(self,text):
        return self.TAG_RE.sub('', text)
    def preprocess_text(self,sen):
        # Removing html tags
        sentence = self.remove_tags(sen)
        # Remove punctuations and numbers
        #sentence = re.sub('[^a-zA-Z]', ' ', sentence)
        # Single character removal
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
        # Removing multiple spaces
        sentence = re.sub(r'\s+', ' ', sentence)
        return sentence
    def masking_error_value(self, filelist, rd_folder_path):
        error_value=[]
        vicinity_with_amsk=[]
        mask="[MASK]"
        for rf in filelist:
            if rf.endswith(".json"):
                try:
                    revision_list = json.load(io.open(os.path.join(rd_folder_path, rf), encoding="utf-8"))    
                    #one_item=revision_list[-1]
                    #old_value=str(one_item[0]['old_value'].strip())
                    #new_value=str(one_item[0]['new_value'].strip())
                    for one_item in revision_list:
                        vicinity=one_item[0]['vicinity']
                        vicinity=remove_markup(str(vicinity))
                        vicinity= ast.literal_eval(vicinity)
                        vicinity=list(filter(None, vicinity))
                        #print(vicinity[0])
                        error_value=one_item[0]['old_value']
                        #print('Before preprocess: ',vicinity)
                        error_value=self.preprocess_text(error_value)
                        #vicinity=preprocess_text(error_value)
                        vicinity=[self.preprocess_text(item) for item in vicinity]
                        print('Before Masking: ', vicinity)
                        print('Error value : ', error_value)
                        error_value=error_value.strip()
                        vicinity=[mask if str(x).strip()==str(error_value) else x for x in vicinity]
                        #vicinity = vicinity.replace(error_value, '**mask**')
                        print('After masking : ',vicinity)
                except Exception as e:
                    print('Exception: ',str(e))
    def prepare_hospital_datasets_finetune(self,clean_data_path, dirty_data_path):
        train_data_rows=[]
        actual_error = pd.DataFrame(columns = ['city', 'state','zip','actual_zip'])
        clean_data=pd.read_csv(clean_data_path)
        dirty_data=pd.read_csv(dirty_data_path)
        clean_data_col=['Address1','City','State','ZipCode','CountyName']
        dirty_data_col=['address_1','city','state','zip','county']
        clean_data = clean_data.applymap(str) # convert whole frame into string
        dirty_data = dirty_data.applymap(str)
        clean_data["ZipCode"] = clean_data["ZipCode"].str.strip()
        dirty_data["zip"] = dirty_data["zip"].str.strip()
        #df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
        #df_obj = dirty_data.select_dtypes(['object'])
        #dirty_data[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
        #df_obj = clean_data.select_dtypes(['object'])
        #clean_data[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
        for dir_col, clean_col in zip(dirty_data_col, clean_data_col):
            dirty_data.loc[dirty_data[dir_col] != clean_data[clean_col], dir_col] = "[MASK]" #replace error value with NaN
        print(dirty_data.zip.values)
        for city1,state1,zip1,zip2 in zip(dirty_data.city,dirty_data.state,dirty_data.zip,clean_data.ZipCode):
            if zip1=="[MASK]":
                actual_error.loc[-1] = [city1, state1,zip1,zip2] #may contain mask in city state
                actual_error.index = actual_error.index + 1  # shifting index
                actual_error = actual_error.sort_index()
            else:
                #print("yes")
                item=[]              
                item.append(city1)
                item.append(state1)
                item.append(zip1)
                train_data_rows.append(item)
        actual_error.to_csv("hospital_zip_actual.csv")
        txt=""
        c=[[' '.join(i)] for i in train_data_rows]
        for sentence in c:
            sentence[0]=sentence[0].replace('[MASK]','')
            txt=txt+ sentence[0]+" ."
        with open("train_bert_hospital.txt", "w") as output:
            output.write(txt)
    def prepare_tax_datasets_finetune(self,clean_data_path, dirty_data_path):
        actual_error = pd.DataFrame(columns = ['city', 'state','actual'])
        train_data_row=[]
        clean_data=pd.read_csv(clean_data_path)
        dirty_data=pd.read_csv(dirty_data_path)
        clean_data_col=['city','state']
        dirty_data_col=['city','state']
        #clean_data_col=clean_data.columns.values
        #dirty_data_col=dirty_data.columns.values
        for dir_col, clean_col in zip(dirty_data_col, clean_data_col):
            dirty_data.loc[dirty_data[dir_col] != clean_data[clean_col], dir_col] = "[MASK]" #replace error value with NaN
        for city1, state1,city2,state2 in zip(dirty_data.city,dirty_data.state,clean_data.city,clean_data.state):
            if city1=="[MASK]" and state1=="[MASK]":
                continue
            elif city1!="[MASK]" and state1!="[MASK]":
                item=[]
                item.append("city")
                item.append(city1)
                item.append("state")
                item.append(state1)
                train_data_row.append(item)
            elif city1!="[MASK]" and state1=="[MASK]":
                actual_error.loc[-1] = [city1, state1,state2]
                actual_error.index = actual_error.index + 1  # shifting index
                actual_error = actual_error.sort_index()
            elif city1=="[MASK]" and state1!="[MASK]":
                actual_error.loc[-1] = [city1, state1,city2]
                actual_error.index = actual_error.index + 1  # shifting index
                actual_error = actual_error.sort_index()
        actual_error.to_csv("tax_city_state_actual.csv")
        txt=""
        c=[[' '.join(i)] for i in train_data_row]
        for sentence in c:
            sentence[0]=sentence[0].replace('[MASK]','')
            txt=txt+ sentence[0]+"."+"\n"
        with open("train_bert_tax.txt", "w") as output:
            output.write(txt)


        #return dirty_data
    def prepare_wiki_datasets_finetune(self, file_list_wiki,rd_folder_path,domain_type):
        train_data_rows=[]
        if domain_type=="location":
            domain_location=['Country', 'COUNTRY', 'country', 'CITY', 'City','city','Location','LOCATION','location','Place','PLACE','place','VENUE','venue','Venue','Town','town','TOWN', 'birth_place','death_place']
        for rf in file_list_wiki:
            if rf.endswith(".json"):
                try:
                    revision_list = json.load(io.open(os.path.join(rd_folder_path, rf), encoding="utf-8"))    
                    one_item=revision_list[-1]
                    if domain_location:
                        if one_item[0]['errored_column'] in domain_location: 
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
                                    train_data_rows.append(row)

                except Exception as e:
                    print('Exception from wiki: ', str(e))
        txt=""
        c=[[' '.join(i)] for i in train_data_rows]
        for sentence in c:
            txt=txt+ sentence[0]+" ."
        with open("train_bert_wiki.txt", "w") as output:
            output.write(txt)
        
        #return train_data_rows
    def prepare_txt_file_training(self, trainlist):
        txt=""
        c=[[' '.join(i)] for i in trainlist]
        for sentence in c:
            txt=txt+ sentence[0]+" ."
        with open("train_bert_wiki.txt", "w") as output:
            output.write(txt)
    def error_correction_BERT_hospital(self, train_data_path, mask_data_path):
        print("Fine tune mask model ")
        happy_roberta = HappyROBERTA()
        word_prediction_args = {
        "batch_size": 1,"epochs": 1,"lr": 5e-5,"adam_epsilon": 1e-8} 
        happy_roberta.init_train_mwp(word_prediction_args)
        happy_roberta.train_mwp(train_data_path)
        pickle.dump(happy_roberta, open("happy_roberto_hos.pickle","wb"))
        print("test mask data ")
        mask_data=pd.read_csv(mask_data_path)
        for city1, state1, zip1, rightzip in zip(mask_data.city,mask_data.state,mask_data.zip,mask_data.actual_zip):
            txt=""
            if city1=="[MASK]":
                city1=""
            if state1=="[MASK]":
                state1=""
            txt=str(city1)+" "+str(state1)+" "+str(zip1)
            
            results=happy_roberta.predict_mask(txt, num_results=3)# no option
            print("Results: ", results, "Actual: ", rightzip)





        
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

if __name__ == "__main__":
    app = BertCorrection()
    wiki_data_path_table="datasets/Table_for_creating_model/"
    filelist = os.listdir(os.path.join(wiki_data_path_table))
    #Split_train_test_datasets_wiki
    train_datasets,test_datasets=app.split_train_test_data(filelist)
    train_wiki_data="bert_datasets/train_bert_wiki.txt"
    #################load tax data##################
    clean_data_path_tax="datasets/tax/clean.csv"
    dirty_data_path_tax="datasets/tax/dirty.csv"
    train_tax_data="bert_datasets/train_bert_tax.txt"
    #################load hospital data#############
    clean_data_path_hos="datasets/hospital/clean.csv"
    dirty_data_path_hos="datasets/hospital/dirty.csv"
    train_hos_data="bert_datasets/train_bert_hospital.txt"
    mask_hospital="bert_datasets/hospital_zip_actual.csv"
    ##########################################
    app.prepare_tax_datasets_finetune(clean_data_path_tax,dirty_data_path_tax)
    #print("Preparing hospital data: ")
    #app.prepare_hospital_datasets_finetune(clean_data_path_hos,dirty_data_path_hos)
    #print("Preparing wiki data: ")
    #app.prepare_wiki_datasets_finetune(filelist,wiki_data_path_table,"location")
    #
    # Error Correction BERT Hospital
    #app.error_correction_BERT_hospital(train_hos_data, mask_hospital)

    