:: Switch path
cd ../

:: TF-IDF
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