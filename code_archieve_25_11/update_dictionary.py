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
#############################################
#test_data_path_domain = os.path.join("performance_final_dictionary","data_error_score.dictionary")
#detected_error_d=pickle.load(open(os.path.join(test_data_path_domain), "rb"))
performance_dictinaries = os.path.join("performance_final_dictionary","data_error_score.dictionary")
if not os.path.exists(performance_dictinaries):
    print("Perfromance matrix  not exits!")
    pickle.dump("results_data_error", open(os.path.join(performance_dictinaries), "wb"))
else:
    print("Perfromance matrix exits!")
    detected_error=pickle.load(open(os.path.join(performance_dictinaries), "rb"))
