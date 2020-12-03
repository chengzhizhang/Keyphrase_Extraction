#  Enhancing Keyphrase Extraction from Academic Articles with their Reference Information





## Dataset
* Semeval2010_244.json #Raw data set of Semeval2010
* KP20K_2000.json #Randomly selected samples from KP20K corpus, the number of which is 2000.

## Source Codes
### Preprocessing 
* a_range.py #Reordering the data in the dataset
* b_data.py  #Stop word filtering, Symbol removal and Stemming.
 
### TF-IDF
* a_tfidf_stem.py #Automatic keyphrase extraction Methods based on TF*IDF
* b_evaluate.py #Calculate the P, R and F1 values of the extraction results
* c_evaluate_key.py #Calculate R values under different conditions with reference information added

### TextRank
* a_textrank_stem.py #Automatic keyphrase extraction Methods based on TextRank
* b_evaluate.py #Calculate the P, R and F1 values of the extraction results
* c_evaluate_key.py #Calculate R values under different conditions with reference information added

### NaiveBayes
* a_dataset_setup.py #The corpus is divided into ten data sets for ten fold cross validation.
* b_corpus_preparing.py #Process the corpus into the format required by KEA.
* b_preparing_keyword.py #Process the keywords into the format required by KEA
* c_merge_key.py #Merge the predicted keywords for each paper into a single file, line by line.
* d_evaluate.py #Calculate the P, R and F1 values of the extraction results
* e_evaluate_key.py #Calculate R values under different conditions with reference information

### CRF
* a_textrank.py #Calculate the TextRank value.
* a_tfidf.py #Calculate the TF*IDF value.
* b_pos_tagging.py #Part-of-speech tagging
* c_features_annotation.py #Annotation: TextRank value, TF*IDF value, position, sequence 5_tag {S,B,M,E,N}, part of speech
* d_dataset_setup.py #The corpus is divided into test set and training set.
* e_features_selection.py #Select the characteristics of the corpus.
* f_tag_prediction.py #Writes the predicted words with {S,B,M,E} tags in line order to the new file.
* g_keyword_generation.py #Combines the extracted labeled words and writes them to the new file.
* h_evaluate.py  #Calculate the P, R and F1 values of the extraction results
* i_evaluate.py #Calculate R values under different conditions with reference information
* template for CRF model


###  References
Chengzhi Zhang, Mengyuan Zhao, Yingyi Zhang. Enhancing Keyphrase Extraction from Academic Articles with their Reference Information. 2020(Under Review) 
