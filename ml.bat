:: Switch path
cd ML

:: create folder
python build_path.py

:: 1. TF-IDF
:: SemEval-2010
:: without refs
python tf_idf.py -m run -dn SemEval-2010 -fd title abstract
python tf_idf.py -m evaluate1 -dn SemEval-2010 -fd title abstract -en 3 5 7 10 -sp ./results/SemEval-2010/tf_idf_no_ref_no_in_tit_abs.csv

:: with refs
python tf_idf.py -m run -dn SemEval-2010 -fd title abstract references
python tf_idf.py -m evaluate1 -dn SemEval-2010 -fd title abstract references -en 3 5 7 10 -sp ./results/SemEval-2010/tf_idf_with_ref_no_in_tit_abs.csv
python tf_idf.py -m evaluate2 -dn SemEval-2010 -fd title abstract references -en 3 5 7 10 -sp ./results/SemEval-2010/tf_idf_with_ref_in_tit_abs.csv

:: KP20k
:: without refs
python tf_idf.py -m run -dn KP20k -fd title abstract
python tf_idf.py -m evaluate1 -dn KP20k -fd title abstract -en 3 5 7 10 -sp ./results/KP20k/tf_idf_no_ref_no_in_tit_abs.csv

:: with refs
python tf_idf.py -m run -dn KP20k -fd title abstract references
python tf_idf.py -m evaluate1 -dn KP20k -fd title abstract references -en 3 5 7 10 -sp ./results/KP20k/tf_idf_with_ref_no_in_tit_abs.csv
python tf_idf.py -m evaluate2 -dn KP20k -fd title abstract references -en 3 5 7 10 -sp ./results/KP20k/tf_idf_with_ref_in_tit_abs.csv

:: LIS-2000
:: without refs
python tf_idf.py -m run -dn LIS-2000 -fd title abstract
python tf_idf.py -m evaluate1 -dn LIS-2000 -fd title abstract -en 3 5 7 10 -sp ./results/LIS-2000/tf_idf_no_ref_no_in_tit_abs.csv

:: with refs
python tf_idf.py -m run -dn LIS-2000 -fd title abstract references
python tf_idf.py -m evaluate1 -dn LIS-2000 -fd title abstract references -en 3 5 7 10 -sp ./results/LIS-2000/tf_idf_with_ref_no_in_tit_abs.csv
python tf_idf.py -m evaluate2 -dn LIS-2000 -fd title abstract references -en 3 5 7 10 -sp ./results/LIS-2000/tf_idf_with_ref_in_tit_abs.csv

:: 2. TextRank
:: SemEval-2010
:: without refs
python textrank.py -m run -dn SemEval-2010 -fd title abstract
python textrank.py -m evaluate1 -dn SemEval-2010 -fd title abstract -en 3 5 7 10 -sp ./results/SemEval-2010/textrank_no_ref_no_in_tit_abs.csv

:: with refs
python textrank.py -m run -dn SemEval-2010 -fd title abstract references
python textrank.py -m evaluate1 -dn SemEval-2010 -fd title abstract references -en 3 5 7 10 -sp ./results/SemEval-2010/textrank_with_ref_no_in_tit_abs.csv
python textrank.py -m evaluate2 -dn SemEval-2010 -fd title abstract references -en 3 5 7 10 -sp ./results/SemEval-2010/textrank_with_ref_in_tit_abs.csv

:: KP20k
:: without refs
python textrank.py -m run -dn KP20k -fd title abstract
python textrank.py -m evaluate1 -dn KP20k -fd title abstract -en 3 5 7 10 -sp ./results/KP20k/textrank_no_ref_no_in_tit_abs.csv

:: with refs
python textrank.py -m run -dn KP20k -fd title abstract references
python textrank.py -m evaluate1 -dn KP20k -fd title abstract references -en 3 5 7 10 -sp ./results/KP20k/textrank_with_ref_no_in_tit_abs.csv
python textrank.py -m evaluate2 -dn KP20k -fd title abstract references -en 3 5 7 10 -sp ./results/KP20k/textrank_with_ref_in_tit_abs.csv

:: LIS-2000
:: without refs
python textrank.py -m run -dn LIS-2000 -fd title abstract
python textrank.py -m evaluate1 -dn LIS-2000 -fd title abstract -en 3 5 7 10 -sp ./results/LIS-2000/textrank_no_ref_no_in_tit_abs.csv

:: with refs
python textrank.py -m run -dn LIS-2000 -fd title abstract references
python textrank.py -m evaluate1 -dn LIS-2000 -fd title abstract references -en 3 5 7 10 -sp ./results/LIS-2000/textrank_with_ref_no_in_tit_abs.csv
python textrank.py -m evaluate2 -dn LIS-2000 -fd title abstract references -en 3 5 7 10 -sp ./results/LIS-2000/textrank_with_ref_in_tit_abs.csv

:: 3. naivebayes
:: SemEval-2010
:: without refs
python naivebayes.py -m data -dn SemEval-2010 -fd title abstract
python naivebayes.py -m train -dn SemEval-2010 -fd title abstract
python naivebayes.py -m test -dn SemEval-2010 -fd title abstract
python naivebayes.py -m evaluate1 -dn SemEval-2010 -fd title abstract -en 3 5 7 10 -sp ./results/SemEval-2010/naivebayes_no_ref_no_in_tit_abs.csv

:: with refs
python naivebayes.py -m data -dn SemEval-2010 -fd title abstract references
python naivebayes.py -m train -dn SemEval-2010 -fd title abstract references
python naivebayes.py -m test -dn SemEval-2010 -fd title abstract references
python naivebayes.py -m evaluate1 -dn SemEval-2010 -fd title abstract references -en 3 5 7 10 -sp ./results/SemEval-2010/naivebayes_with_ref_no_in_tit_abs.csv
python naivebayes.py -m evaluate2 -dn SemEval-2010 -fd title abstract references -en 3 5 7 10 -sp ./results/SemEval-2010/naivebayes_with_ref_in_tit_abs.csv

:: KP20k
:: without refs
python naivebayes.py -m data -dn KP20k -fd title abstract
python naivebayes.py -m train -dn KP20k -fd title abstract
python naivebayes.py -m test -dn KP20k -fd title abstract
python naivebayes.py -m evaluate1 -dn KP20k -fd title abstract -en 3 5 7 10 -sp ./results/KP20k/naivebayes_no_ref_no_in_tit_abs.csv

:: with refs
python naivebayes.py -m data -dn KP20k -fd title abstract references
python naivebayes.py -m train -dn KP20k -fd title abstract references
python naivebayes.py -m test -dn KP20k -fd title abstract references
python naivebayes.py -m evaluate1 -dn KP20k -fd title abstract references -en 3 5 7 10 -sp ./results/KP20k/naivebayes_with_ref_no_in_tit_abs.csv
python naivebayes.py -m evaluate2 -dn KP20k -fd title abstract references -en 3 5 7 10 -sp ./results/KP20k/naivebayes_with_ref_in_tit_abs.csv

:: LIS-2000
:: without refs
python naivebayes.py -m data -dn LIS-2000 -fd title abstract
python naivebayes.py -m train -dn LIS-2000 -fd title abstract
python naivebayes.py -m test -dn LIS-2000 -fd title abstract
python naivebayes.py -m evaluate1 -dn LIS-2000 -fd title abstract -en 3 5 7 10 -sp ./results/LIS-2000/naivebayes_no_ref_no_in_tit_abs.csv

:: with refs
python naivebayes.py -m data -dn LIS-2000 -fd title abstract references
python naivebayes.py -m train -dn LIS-2000 -fd title abstract references
python naivebayes.py -m test -dn LIS-2000 -fd title abstract references
python naivebayes.py -m evaluate1 -dn LIS-2000 -fd title abstract references -en 3 5 7 10 -sp ./results/LIS-2000/naivebayes_with_ref_no_in_tit_abs.csv
python naivebayes.py -m evaluate2 -dn LIS-2000 -fd title abstract references -en 3 5 7 10 -sp ./results/LIS-2000/naivebayes_with_ref_in_tit_abs.csv

:: 4. Crf
:: SemEval-2010
:: without refs
python crf.py -m data -dn SemEval-2010 -fd title abstract
python crf.py -m train -dn SemEval-2010 -fd title abstract
python crf.py -m test -dn SemEval-2010 -fd title abstract
python crf.py -m evaluate1 -dn SemEval-2010 -fd title abstract -sp ./results/SemEval-2010/crf_no_ref_no_in_tit_abs.csv

:: with refs
python crf.py -m data -dn SemEval-2010 -fd title abstract references
python crf.py -m train -dn SemEval-2010 -fd title abstract references
python crf.py -m test -dn SemEval-2010 -fd title abstract references
python crf.py -m evaluate1 -dn SemEval-2010 -fd title abstract references -sp ./results/SemEval-2010/crf_with_ref_no_in_tit_abs.csv
python crf.py -m evaluate2 -dn SemEval-2010 -fd title abstract references -sp ./results/SemEval-2010/crf_with_ref_in_tit_abs.csv

:: KP20k
:: without refs
python crf.py -m data -dn KP20k -fd title abstract
python crf.py -m train -dn KP20k -fd title abstract
python crf.py -m test -dn KP20k -fd title abstract
python crf.py -m evaluate1 -dn KP20k -fd title abstract -sp ./results/KP20k/crf_no_ref_no_in_tit_abs.csv

:: with refs
python crf.py -m data -dn KP20k -fd title abstract references
python crf.py -m train -dn KP20k -fd title abstract references
python crf.py -m test -dn KP20k -fd title abstract references
python crf.py -m evaluate1 -dn KP20k -fd title abstract references -sp ./results/KP20k/crf_with_ref_no_in_tit_abs.csv
python crf.py -m evaluate2 -dn KP20k -fd title abstract references -sp ./results/KP20k/crf_with_ref_in_tit_abs.csv

:: LIS-2000
:: without refs
python crf.py -m data -dn LIS-2000 -fd title abstract
python crf.py -m train -dn LIS-2000 -fd title abstract
python crf.py -m test -dn LIS-2000 -fd title abstract
python crf.py -m evaluate1 -dn LIS-2000 -fd title abstract -sp ./results/LIS-2000/crf_no_ref_no_in_tit_abs.csv

:: with refs
python crf.py -m data -dn LIS-2000 -fd title abstract referencess
python crf.py -m train -dn LIS-2000 -fd title abstract references
python crf.py -m test -dn LIS-2000 -fd title abstract references
python crf.py -m evaluate1 -dn LIS-2000 -fd title abstract references -sp ./results/LIS-2000/crf_with_ref_no_in_tit_abs.csv
python crf.py -m evaluate2 -dn LIS-2000 -fd title abstract references -sp ./results/LIS-2000/crf_with_ref_in_tit_abs.csv
