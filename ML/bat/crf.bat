:: Switch path
cd ../

:: Crf
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