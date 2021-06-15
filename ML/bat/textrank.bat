:: Switch path
cd ../

:: TextRank
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