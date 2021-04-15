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
import dataset
import wikitextparser as wtp
from pprint import pprint
from wikitextparser import remove_markup, parse
import datetime
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
             
    def extract_revisions(self, wikipedia_dumps_folder):
        """
        This method takes the folder path of Wikipedia page revision history dumps and extracts infobox/table revision data.
        """
        self.rd_folder_path = os.path.join(wikipedia_dumps_folder, "revision-data")
        if not os.path.exists(self.rd_folder_path):
            os.mkdir(self.rd_folder_path)
        compressed_dumps_list = [df for df in os.listdir(wikipedia_dumps_folder) if df.endswith(".7z")]
        for file_name in compressed_dumps_list:
            compressed_dump_file_path = os.path.join(wikipedia_dumps_folder, file_name)
            dump_file_name, _ = os.path.splitext(os.path.basename(compressed_dump_file_path))
            self.rdd_folder_path = os.path.join(self.rd_folder_path, dump_file_name)
            if not os.path.exists(self.rdd_folder_path):
                os.mkdir(self.rdd_folder_path)
            else:
                continue
            archive = py7zr.SevenZipFile(compressed_dump_file_path, mode="r")
            archive.extractall(path=wikipedia_dumps_folder)
            archive.close()
            decompressed_dump_file_path = os.path.join(wikipedia_dumps_folder, dump_file_name)
            decompressed_dump_file = io.open(decompressed_dump_file_path, "r", encoding="utf-8")
            page_text = ""
            for i,line in enumerate(decompressed_dump_file):
                line = line.strip()
                if line == "<page>":
                    page_text = ""
                page_text += "\n" + line
                if line == "</page>":
                    page_tree = bs4.BeautifulSoup(page_text, "html.parser")
                    self.page_folder=str(page_tree.id.text)
                    filelist1 = os.listdir('/content/drive/My Drive/datasets/revision-data/03-09-2020')
                    #if int(self.page_folder)<=5030548 or sys.getsizeof(page_text)> 5000000:
                    if sys.getsizeof(page_text)> 500000000:
                        print('Page size: ', sys.getsizeof(page_text), ' byte')
                        #print(self.page_folder, ':Page already parsed')
                        #print()
                        continue
                    else:
                        print(self.page_folder, 'is processing now')
                        print('Page size: ', sys.getsizeof(page_text), ' byte')
                        print('Start Time', datetime.datetime.now())
                        total_infobox_count=0
                        total_table_count=0
                        for revision_tag in page_tree.find_all("revision"):
                            self.revision_id_parent="root"
                            self.revision_id_current=revision_tag.find("id").text
                            try:
                                self.revision_id_parent=revision_tag.find("parentid").text
                            except Exception as e:
                                print('Exception: Parent Id: ', str(e))
                            revision_text = revision_tag.find("text").text
                            self.code =mwparserfromhell.parse(revision_text)
                            self.table=self.code.filter_tags(matches=lambda node: node.tag=="table")
                            revision_table_count=self.table_parsing()
                            #revision_infobox_count=self.infobox_parsing()
                            #total_infobox_count=total_infobox_count+revision_infobox_count
                            total_table_count=total_table_count+revision_table_count
                        #print("The total number of infobox in this page(with revision): {}".format(total_infobox_count))
                        print(self.page_folder, ' of processing is finished')
                        print('End time', datetime.datetime.now())
                        print("The total number of table in this page(with revision): {}".format(total_table_count))            
            decompressed_dump_file.close()
            os.remove(decompressed_dump_file_path)
    def infobox_parsing(self):
        """
        This method will extract all infobox templates with revision
        """
        infobox_count=0
        templates = self.code.filter_templates()
        for temp in templates:
            json_list=[]
            if "Infobox" in temp.name:
                self.revision_page_folder_path=os.path.join(self.rdd_folder_path,self.page_folder)
                if not os.path.exists(self.revision_page_folder_path):
                    os.mkdir(self.revision_page_folder_path)
                infobox_folder=remove_markup(str(temp.name))
                infobox_folder= re.sub('[^a-zA-Z0-9\n\.]', ' ', (str(infobox_folder)).lower())
                revision_infobox_folder_path=os.path.join(self.revision_page_folder_path,infobox_folder)
                if not os.path.exists(revision_infobox_folder_path):
                    os.mkdir(revision_infobox_folder_path)
                json_list.append(str(temp))
                json.dump(json_list, open(os.path.join(revision_infobox_folder_path, self.revision_id_parent + '_' + self.revision_id_current + ".json"), "w"))
                print(temp.name)
                infobox_count=infobox_count+1
        return infobox_count
    def table_parsing(self):
        """
        This method will extract all table templates with revision
        """
        table_count=0
        if self.table:                       
            for tebil in self.table:
                json_list=[]
                try:
                    table_caption = wtp.parse(str(tebil)).tables[0].caption
                    table_folder_name=remove_markup(str(table_caption))
                    table_folder_name=table_folder_name.lower()
                    table_folder_name=table_folder_name.strip()
                except Exception as e:
                    print('Exception: table folder name or out of list in table', str(e))
                    #print(str(e))
                    continue   
                
                if table_caption:
                  try:
                      self.revision_page_folder_path=os.path.join('/content/drive/My Drive/datasets/revision-data/03-09-2020',self.page_folder)
                      if not os.path.exists(self.revision_page_folder_path):
                          os.mkdir(self.revision_page_folder_path)
                      table_folder_name=table_folder_name.strip('\n')
                      revision_table_folder_path=os.path.join(self.revision_page_folder_path,table_folder_name)
                      revision_table_folder_path=revision_table_folder_path.strip()
                      if not os.path.exists(revision_table_folder_path):
                          os.mkdir(revision_table_folder_path)
                  except Exception as e:
                      print('Exception: revision table folder', str(e))
                      continue
                  table_count=table_count+1
                  json_list.append(str(tebil))
                  json.dump(json_list, open(os.path.join(revision_table_folder_path, self.revision_id_parent + '_' + self.revision_id_current + ".json"), "w"))
                  print('Table caption: ', table_folder_name)
                  table_count=table_count+1
                                     
        return table_count
    def make_list_from_sorted_json(self, revision_data_folder):
        rd_folder_path = revision_data_folder
        list_from_sorted_json_path=os.path.join(revision_data_folder,'all_revision_list_per_table_or_infobox')
        if not os.path.isdir(list_from_sorted_json_path):
            os.mkdir(list_from_sorted_json_path)
        for folder in os.listdir(rd_folder_path): #initial archieve folder
            rdd_folder_path=os.path.join(rd_folder_path,folder)
            if os.path.isdir(os.path.join(rd_folder_path, folder)): #datasets/revision-data/archieve-foldername
                for nested_folder in os.listdir(os.path.join(rd_folder_path,folder)):
                    print(nested_folder)
                    page_folder=os.path.join(rdd_folder_path,nested_folder) #datasets/revision-data/archieve-foldername/page_folder
                    print(page_folder)
                    if os.path.isdir(os.path.join(rdd_folder_path, nested_folder)):
                        for nested_nested_folder in os.listdir(os.path.join(rdd_folder_path,nested_folder)):
                            revision_list=[]
                            print(nested_nested_folder)
                            if os.path.isdir(os.path.join(page_folder, nested_nested_folder)):
                                filelist = os.listdir(os.path.join(page_folder, nested_nested_folder))
                                filelist = sorted(filelist,key=lambda x: int(os.path.splitext(x)[0]))#sort the json file list
                                for rf in filelist:
                                    print(rf)
                                    if rf.endswith(".json"):
                                        try:
                                            revision_list.append(json.load(io.open(os.path.join(page_folder, nested_nested_folder, rf), encoding="utf-8")))
                                        except:
                                            continue
                                data_revision_list=json.dumps(revision_list)
                                json.dump(data_revision_list, open(os.path.join(list_from_sorted_json_path, "%s.json") %nested_nested_folder, "w"))
                            else:
                                print('The path is not found')
                    else:
                        print('The path is not found')
            else:
                print('The path is not found')
        
if __name__ == "__main__":
    app = WikiErrorCorrection()
    app.extract_revisions(wikipedia_dumps_folder="datasets")
    #app.make_list_from_sorted_json(revision_data_folder="datasets/revision-data")
