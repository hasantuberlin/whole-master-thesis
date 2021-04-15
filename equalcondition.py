if repaired_e and  repaired_f and repaired_e==repaired_f:
    result_dictionary_fasttext_edit[i,j]=repaired_e
elif repaired_e and repaired_f and model_confidence>=0.99:
    result_dictionary_fasttext_edit[i,j]=repaired_f
elif (repaired_e==error_value or repaired_e=="None" or not repaired_e ) and repaired_f:
    if model_confidence>=0.99:
        result_dictionary_fasttext_edit[i,j]=repaired_f
elif repaired_e:
    if repaired_e==error_value or repaired_e=="None":
        continue
    else:
        result_dictionary_fasttext_edit[i,j]=repaired_e