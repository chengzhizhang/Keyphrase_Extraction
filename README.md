#  Enhancing Keyphrase Extraction from Academic Articles with their Reference Information


## Overview
<b>Dataset and source code for paper "Enhancing Keyphrase Extraction from Academic Articles with their Reference Information".</b>

The research content of this project is to analyze the impact 
of the introduction of reference title in scientific literature 
on the effect of keyword extraction. This project uses three 
datasets: <b>SemEval-2010</b>, <b>PubMed</b> and <b>LIS-2000</b>, which are located 
in the dataset folder. At the same time, we use two unsupervised 
methods: <b>TF-IDF</b> and <b>TextRank</b>, and three supervised learning methods:
<b>Naïve Bayes</b>, <b>CRF</b> and <b>BiLSTM-CRF</b>. The first four are traditional keywords 
extraction methods, located in the folder <b>ML</b>, and the last one is deep 
learning method, located in the folder <b>DL</b>.
### Directory structure
<pre>Keyphrase_Extraction               Root directory
├─dl.bat                           Batch commands to run deep learning model
├─ml.bat                           Batch commands to run traditional models
│ 
├─Dataset                          Experimental datasets
│  ├─SemEval-2010                  Contains 244 scientific papers 
│  ├─PubMed                        Contains 1316 scientific papers
│  └─LIS-2000                      Contains 2000 scientific papers
│ 
├─DL                               Source code of the deep learning model
│  ├─build_path.py                 Create file paths for saving preprocessed data
│  ├─crf.py                       Source code of CRF algorithm implementation (Use pytorch framework)
│  ├─main.py                       The main function of running the program
│  ├─model.py                      Source code of BiLSTM-CRF model
│  ├─preprocess.py                 Source code of preprocessing function
│  ├─textrank.py                   Source code of TextRank algorithm implementation.
│  ├─tf_idf.py                     Source code of TF-IDF algorithm implementation.
│  ├─utils.py                      Some auxiliary functions
│  ├─models                        Parameter configuration of deep learning models
│  └─datas
│     └─tags                       Label settings for sequence labeling
│ 
├─ML                               Source code of the traditional models
│  ├─build_path.py                 Create file paths for saving preprocessed data
│  ├─configs.py                    Path configuration file
│  ├─crf.py                        Source code of CRF algorithm implementation(Use CRF++ Toolkit)
│  ├─evaluate.py                   Surce code for result evaluation
│  ├─naivebayes.py                 Source code of Naïve Bayes algorithm implementation(Use KEA-3.0 Toolkit)
│  ├─preprocessing.py              Source code of preprocessing function
│  ├─textrank.py                   Source code of TextRank algorithm implementation
│  ├─tf_idf.py                     Source code of TF-IDF algorithm implementation
│  ├─utils.py                      Some auxiliary functions
│  ├─CRF++                         CRF++ Toolkit
│  └─KEA-3.0                       KEA-3.0 Toolkit
│
└─README.md
</pre>

## Dataset Description

<b>The dataset includes the following three json files:</b>
<li> SemEval-2010: SemEval-2010 Task 5 dataset, it contains 244 scientific papers and can be visited at: 
<a href="https://semeval2.fbk.eu/semeval2.php?location=data">https://semeval2.fbk.eu/semeval2.php?location=data</a>. 
<li> PubMed: Contains 1316 scientific papers from PubMed 
(<a href="https://github.com/boudinfl/ake-datasets/tree/master/datasets/PubMed">https://github.com/boudinfl/ake-datasets/tree/master/datasets/PubMed</a>).
<li> LIS-2000: Contains 2000 scientific papers from journals in Library and Information Science (LIS).

<b>Each line of the json file includes: </b>
<li>title (T): The title of the paper.
<li>abstract (A): The abstract of the paper.
<li>introduction (I): The introduction of the paper.
<li>conclusion (C): The conclusion of the paper.
<li>body1 (Fp): The first sentence of each paragraph.
<li>body2 (Lp): The last sentence of each paragraph.
<li>full_text (F): The full text of the paper.
<li>references (R): references list and only the title of each reference is provided.
<li>keywords (K): the keywords of the paper and these keywords were annotated manually.

## Quick Start
In order to facilitate the reproduction of the experimental results, 
the project uses bat batch command to run the program uniformly 
(only in Windows Environment). The <b>dl.bat</b> file is the batch command 
to run the deep learning model, and the <b>ml.bat</b> file is the batch command 
to run the traditional algorithm.

### How does it work?
In the Windows environment, use the key combination <b>Win + R</b> and enter <b>cmd</b>
to open the <b>DOS</b> command box, and switch to the project's root directory 
(Keyphrase_Extraction). Then input <b>dl.bat</b>, that is, run deep learning model 
to get the result of keyword extraction; Enter <b>ml.bat</b> to run traditional 
algorithm to get keywords Extract the results.

## Experimental results
The following figures show that the influence of reference information on keyphrase extraction results of TF*IDF, TextRank, NB, CRF and BiLSTM-CRF.
<br/><br/>
Table 1: Keyphrase extraction performance of multiple corpora constructed using different logical structure texts on the dataset of SemEval-2010
<img src="https://chengzhizhang.github.io/images/img_1.png" alt="Table1"/>
<br/><br/>
Table 2: Keyphrase extraction performance of multiple corpora constructed using different logical structure texts on the dataset of PubMed
<img src="https://chengzhizhang.github.io/images/img_2.png" alt="Table2"/>
<br/><br/>
Table 3: Keyphrase extraction performance of multiple corpora constructed using different logical structure texts on the dataset of LIS-2000
<img src="https://chengzhizhang.github.io/images/img_3.png" alt="Table3"/>
    

<b>Note</b>: The yellow, green and blue bold fonts in the table represent the largest of the P, R and F<sub>1</sub> value obtained from different corpora using the same model, respectively.

## Dependency packages
Before running this project, check that the following Python packages are 
included in your runtime environment.

<li>pytorch 1.7.1

<li>nltk 3.5

<li>numpy 1.19.2

<li>pandas 1.1.3

<li>tqdm 4.50.2


## Citation
Please cite the following paper if you use this code and dataset in your work.
    
>Chengzhi Zhang, Lei Zhao, Mengyuan Zhao, Yingyi Zhang. Enhancing Keyphrase Extraction from Academic Articles with their Reference Information. *Scientometrics*, 2022, 127(2): 703–731. [[doi]](https://doi.org/10.1007/s11192-021-04230-4)  [[arXiv]](http://arxiv.org/abs/2111.14106)
