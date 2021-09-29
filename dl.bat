:: Switch path
cd DL

:: create folder
python build_path.py

:: SemEval-2010
:: POS LEN WFOF WFF WFR IWOT IWOR TI TR
:: TA
python main.py -m train -dn SemEval-2010 -fd title abstract -fs POS LEN WFOF WFF WFR IWOT IWOR TI TR
python main.py -m test -dn SemEval-2010 -fd title abstract -fs POS LEN WFOF WFF WFR IWOT IWOR TI TR
:: TAI
python main.py -m train -dn SemEval-2010 -fd title abstract introduction -fs POS LEN WFOF WFF WFR IWOT IWOR TI TR
python main.py -m test -dn SemEval-2010 -fd title abstract introduction -fs POS LEN WFOF WFF WFR IWOT IWOR TI TR
:: TAC
python main.py -m train -dn SemEval-2010 -fd title abstract conclusion -fs POS LEN WFOF WFF WFR IWOT IWOR TI TR
python main.py -m test -dn SemEval-2010 -fd title abstract conclusion -fs POS LEN WFOF WFF WFR IWOT IWOR TI TR
:: TAFp
python main.py -m train -dn SemEval-2010 -fd title abstract body1 -fs POS LEN WFOF WFF WFR IWOT IWOR TI TR
python main.py -m test -dn SemEval-2010 -fd title abstract body1 -fs POS LEN WFOF WFF WFR IWOT IWOR TI TR
:: TALp
python main.py -m train -dn SemEval-2010 -fd title abstract body2 -fs POS LEN WFOF WFF WFR IWOT IWOR TI TR
python main.py -m test -dn SemEval-2010 -fd title abstract body2 -fs POS LEN WFOF WFF WFR IWOT IWOR TI TR
:: TAR
python main.py -m train -dn SemEval-2010 -fd title abstract references -fs POS LEN WFOF WFF WFR IWOT IWOR TI TR
python main.py -m test -dn SemEval-2010 -fd title abstract references -fs POS LEN WFOF WFF WFR IWOT IWOR TI TR
:: TAF
python main.py -m train -dn SemEval-2010 -fd title abstract full_text -fs POS LEN WFOF WFF WFR IWOT IWOR TI TR
python main.py -m test -dn SemEval-2010 -fd title abstract full_text -fs POS LEN WFOF WFF WFR IWOT IWOR TI TR
:: TAFR
python main.py -m train -dn SemEval-2010 -fd title abstract full_text references -fs POS LEN WFOF WFF WFR IWOT IWOR TI TR
python main.py -m test -dn SemEval-2010 -fd title abstract full_text references -fs POS LEN WFOF WFF WFR IWOT IWOR TI TR
:: TAICFpLp
python main.py -m train -dn SemEval-2010 -fd title abstract introduction conclusion body1 body2 -fs POS LEN WFOF WFF WFR IWOT IWOR TI TR
python main.py -m test -dn SemEval-2010 -fd title abstract introduction conclusion body1 body2 -fs POS LEN WFOF WFF WFR IWOT IWOR TI TR
:: TAICFpLpR
python main.py -m train -dn SemEval-2010 -fd title abstract introduction conclusion body1 body2 references -fs POS LEN WFOF WFF WFR IWOT IWOR TI TR
python main.py -m test -dn SemEval-2010 -fd title abstract introduction conclusion body1 body2 references -fs POS LEN WFOF WFF WFR IWOT IWOR TI TR

:: PubMed
:: IWOT TI TR
:: TA
python main.py -m train -dn PubMed -fd title abstract -fs IWOT TI TR
python main.py -m test -dn PubMed -fd title abstract -fs IWOT TI TR
:: TAI
python main.py -m train -dn PubMed -fd title abstract introduction -fs IWOT TI TR
python main.py -m test -dn PubMed -fd title abstract introduction -fs IWOT TI TR
:: TAC
python main.py -m train -dn PubMed -fd title abstract conclusion -fs IWOT TI TR
python main.py -m test -dn PubMed -fd title abstract conclusion -fs IWOT TI TR
:: TAFp
python main.py -m train -dn PubMed -fd title abstract body1 -fs IWOT TI TR
python main.py -m test -dn PubMed -fd title abstract body1 -fs IWOT TI TR
:: TALp
python main.py -m train -dn PubMed -fd title abstract body2 -fs IWOT TI TR
python main.py -m test -dn PubMed -fd title abstract body2 -fs IWOT TI TR
:: TAR
python main.py -m train -dn PubMed -fd title abstract references -fs IWOT TI TR
python main.py -m test -dn PubMed -fd title abstract references -fs IWOT TI TR
:: TAF
python main.py -m train -dn PubMed -fd title abstract full_text -fs IWOT TI TR
python main.py -m test -dn PubMed -fd title abstract full_text -fs IWOT TI TR
:: TAFR
python main.py -m train -dn PubMed -fd title abstract full_text references -fs IWOT TI TR
python main.py -m test -dn PubMed -fd title abstract full_text references -fs IWOT TI TR
:: TAICFpLp
python main.py -m train -dn PubMed -fd title abstract introduction conclusion body1 body2 -fs IWOT TI TR
python main.py -m test -dn PubMed -fd title abstract introduction conclusion body1 body2 -fs IWOT TI TR
:: TAICFpLpR
python main.py -m train -dn PubMed -fd title abstract introduction conclusion body1 body2 references -fs IWOT TI TR
python main.py -m test -dn PubMed -fd title abstract introduction conclusion body1 body2 references -fs IWOT TI TR

:: LIS-2000
:: POS LEN WFOF WFR IWOT IWOR TI TR
:: TA
python main.py -m train -dn LIS-2000 -fd title abstract -fs POS LEN WFOF WFR IWOT IWOR TI TR
python main.py -m test -dn LIS-2000 -fd title abstract -fs POS LEN WFOF WFR IWOT IWOR TI TR
:: TAI
python main.py -m train -dn LIS-2000 -fd title abstract introduction -fs POS LEN WFOF WFR IWOT IWOR TI TR
python main.py -m test -dn LIS-2000 -fd title abstract introduction -fs POS LEN WFOF WFR IWOT IWOR TI TR
:: TAC
python main.py -m train -dn LIS-2000 -fd title abstract conclusion -fs POS LEN WFOF WFR IWOT IWOR TI TR
python main.py -m test -dn LIS-2000 -fd title abstract conclusion -fs POS LEN WFOF WFR IWOT IWOR TI TR
:: TAFp
python main.py -m train -dn LIS-2000 -fd title abstract body1 -fs POS LEN WFOF WFR IWOT IWOR TI TR
python main.py -m test -dn LIS-2000 -fd title abstract body1 -fs POS LEN WFOF WFR IWOT IWOR TI TR
:: TALp
python main.py -m train -dn LIS-2000 -fd title abstract body2 -fs POS LEN WFOF WFR IWOT IWOR TI TR
python main.py -m test -dn LIS-2000 -fd title abstract body2 -fs POS LEN WFOF WFR IWOT IWOR TI TR
:: TAR
python main.py -m train -dn LIS-2000 -fd title abstract references -fs POS LEN WFOF WFR IWOT IWOR TI TR
python main.py -m test -dn LIS-2000 -fd title abstract references -fs POS LEN WFOF WFR IWOT IWOR TI TR
:: TAF
python main.py -m train -dn LIS-2000 -fd title abstract full_text -fs POS LEN WFOF WFR IWOT IWOR TI TR
python main.py -m test -dn LIS-2000 -fd title abstract full_text -fs POS LEN WFOF WFR IWOT IWOR TI TR
:: TAFR
python main.py -m train -dn LIS-2000 -fd title abstract full_text references -fs POS LEN WFOF WFR IWOT IWOR TI TR
python main.py -m test -dn LIS-2000 -fd title abstract full_text references -fs POS LEN WFOF WFR IWOT IWOR TI TR
:: TAICFpLp
python main.py -m train -dn LIS-2000 -fd title abstract introduction conclusion body1 body2 -fs POS LEN WFOF WFR IWOT IWOR TI TR
python main.py -m test -dn LIS-2000 -fd title abstract introduction conclusion body1 body2 -fs POS LEN WFOF WFR IWOT IWOR TI TR
:: TAICFpLpR
python main.py -m train -dn LIS-2000 -fd title abstract introduction conclusion body1 body2 references -fs POS LEN WFOF WFR IWOT IWOR TI TR
python main.py -m test -dn LIS-2000 -fd title abstract introduction conclusion body1 body2 references -fs POS LEN WFOF WFR IWOT IWOR TI TR
