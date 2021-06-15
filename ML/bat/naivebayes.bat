:: Switch path
cd ../

:: naivebayes
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
