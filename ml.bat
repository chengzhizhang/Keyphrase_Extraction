:: Switch path
cd ML

:: create folder
python build_path.py

:: 1. TF-IDF
:: SemEval-2010
:: TA
python tf_idf.py -m run -dn SemEval-2010 -fd title abstract
python tf_idf.py -m evaluate -dn SemEval-2010 -fd title abstract -en 3 5 7 10
:: TAI
python tf_idf.py -m run -dn SemEval-2010 -fd title abstract introduction
python tf_idf.py -m evaluate -dn SemEval-2010 -fd title abstract introduction -en 3 5 7 10
:: TAC
python tf_idf.py -m run -dn SemEval-2010 -fd title abstract conclusion
python tf_idf.py -m evaluate -dn SemEval-2010 -fd title abstract conclusion -en 3 5 7 10
:: TAFp
python tf_idf.py -m run -dn SemEval-2010 -fd title abstract body1
python tf_idf.py -m evaluate -dn SemEval-2010 -fd title abstract body1 -en 3 5 7 10
:: TALp
python tf_idf.py -m run -dn SemEval-2010 -fd title abstract body2
python tf_idf.py -m evaluate -dn SemEval-2010 -fd title abstract body2 -en 3 5 7 10
:: TAR
python tf_idf.py -m run -dn SemEval-2010 -fd title abstract references
python tf_idf.py -m evaluate -dn SemEval-2010 -fd title abstract references -en 3 5 7 10
:: TAF
python tf_idf.py -m run -dn SemEval-2010 -fd title abstract full_text
python tf_idf.py -m evaluate -dn SemEval-2010 -fd title abstract full_text -en 3 5 7 10
:: TAFR
python tf_idf.py -m run -dn SemEval-2010 -fd title abstract full_text references
python tf_idf.py -m evaluate -dn SemEval-2010 -fd title abstract full_text references -en 3 5 7 10
:: TAICFpLp
python tf_idf.py -m run -dn SemEval-2010 -fd title abstract introduction conclusion body1 body2
python tf_idf.py -m evaluate -dn SemEval-2010 -fd title abstract introduction conclusion body1 body2 -en 3 5 7 10
:: TAICFpLpR
python tf_idf.py -m run -dn SemEval-2010 -fd title abstract introduction conclusion body1 body2 references
python tf_idf.py -m evaluate -dn SemEval-2010 -fd title abstract introduction conclusion body1 body2 references -en 3 5 7 10

:: PubMed
:: TA
python tf_idf.py -m run -dn PubMed -fd title abstract
python tf_idf.py -m evaluate -dn PubMed -fd title abstract -en 3 5 7 10
:: TAI
python tf_idf.py -m run -dn PubMed -fd title abstract introduction
python tf_idf.py -m evaluate -dn PubMed -fd title abstract introduction -en 3 5 7 10
:: TAC
python tf_idf.py -m run -dn PubMed -fd title abstract conclusion
python tf_idf.py -m evaluate -dn PubMed -fd title abstract conclusion -en 3 5 7 10
:: TAFp
python tf_idf.py -m run -dn PubMed -fd title abstract body1
python tf_idf.py -m evaluate -dn PubMed -fd title abstract body1 -en 3 5 7 10
:: TALp
python tf_idf.py -m run -dn PubMed -fd title abstract body2
python tf_idf.py -m evaluate -dn PubMed -fd title abstract body2 -en 3 5 7 10
:: TAR
python tf_idf.py -m run -dn PubMed -fd title abstract references
python tf_idf.py -m evaluate -dn PubMed -fd title abstract references -en 3 5 7 10
:: TAF
python tf_idf.py -m run -dn PubMed -fd title abstract full_text
python tf_idf.py -m evaluate -dn PubMed -fd title abstract full_text -en 3 5 7 10
:: TAFR
python tf_idf.py -m run -dn PubMed -fd title abstract full_text references
python tf_idf.py -m evaluate -dn PubMed -fd title abstract full_text references -en 3 5 7 10
:: TAICFpLp
python tf_idf.py -m run -dn PubMed -fd title abstract introduction conclusion body1 body2
python tf_idf.py -m evaluate -dn PubMed -fd title abstract introduction conclusion body1 body2 -en 3 5 7 10
:: TAICFpLp
python tf_idf.py -m run -dn PubMed -fd title abstract introduction conclusion body1 body2 references
python tf_idf.py -m evaluate -dn PubMed -fd title abstract introduction conclusion body1 body2 references -en 3 5 7 10

:: LIS-2000
:: TA
python tf_idf.py -m run -dn LIS-2000 -fd title abstract
python tf_idf.py -m evaluate -dn LIS-2000 -fd title abstract -en 3 5 7 10
:: TAI
python tf_idf.py -m run -dn LIS-2000 -fd title abstract introduction
python tf_idf.py -m evaluate -dn LIS-2000 -fd title abstract introduction -en 3 5 7 10
:: TAC
python tf_idf.py -m run -dn LIS-2000 -fd title abstract conclusion
python tf_idf.py -m evaluate -dn LIS-2000 -fd title abstract conclusion -en 3 5 7 10
:: TAFp
python tf_idf.py -m run -dn LIS-2000 -fd title abstract body1
python tf_idf.py -m evaluate -dn LIS-2000 -fd title abstract body1 -en 3 5 7 10
:: TALp
python tf_idf.py -m run -dn LIS-2000 -fd title abstract body2
python tf_idf.py -m evaluate -dn LIS-2000 -fd title abstract body2 -en 3 5 7 10
:: TAR
python tf_idf.py -m run -dn LIS-2000 -fd title abstract references
python tf_idf.py -m evaluate -dn LIS-2000 -fd title abstract references -en 3 5 7 10
:: TAF
python tf_idf.py -m run -dn LIS-2000 -fd title abstract full_text
python tf_idf.py -m evaluate -dn LIS-2000 -fd title abstract full_text -en 3 5 7 10
:: TAFR
python tf_idf.py -m run -dn LIS-2000 -fd title abstract full_text references
python tf_idf.py -m evaluate -dn LIS-2000 -fd title abstract full_text references -en 3 5 7 10
:: TAICFpLp
python tf_idf.py -m run -dn LIS-2000 -fd title abstract introduction conclusion body1 body2
python tf_idf.py -m evaluate -dn LIS-2000 -fd title abstract introduction conclusion body1 body2 -en 3 5 7 10
:: TAICFpLpR
python tf_idf.py -m run -dn LIS-2000 -fd title abstract introduction conclusion body1 body2 references
python tf_idf.py -m evaluate -dn LIS-2000 -fd title abstract introduction conclusion body1 body2 references -en 3 5 7 10

:: 2. TextRank
:: SemEval-2010
:: TA
python textrank.py -m run -dn SemEval-2010 -fd title abstract
python textrank.py -m evaluate -dn SemEval-2010 -fd title abstract -en 3 5 7 10
:: TAI
python textrank.py -m run -dn SemEval-2010 -fd title abstract introduction
python textrank.py -m evaluate -dn SemEval-2010 -fd title abstract introduction -en 3 5 7 10
:: TAC
python textrank.py -m run -dn SemEval-2010 -fd title abstract conclusion
python textrank.py -m evaluate -dn SemEval-2010 -fd title abstract conclusion -en 3 5 7 10
:: TAFp
python textrank.py -m run -dn SemEval-2010 -fd title abstract body1
python textrank.py -m evaluate -dn SemEval-2010 -fd title abstract body1 -en 3 5 7 10
:: TAFp
python textrank.py -m run -dn SemEval-2010 -fd title abstract body2
python textrank.py -m evaluate -dn SemEval-2010 -fd title abstract body2 -en 3 5 7 10
:: TAR
python textrank.py -m run -dn SemEval-2010 -fd title abstract references
python textrank.py -m evaluate -dn SemEval-2010 -fd title abstract references -en 3 5 7 10
:: TAF
python textrank.py -m run -dn SemEval-2010 -fd title abstract full_text
python textrank.py -m evaluate -dn SemEval-2010 -fd title abstract full_text -en 3 5 7 10
:: TAFR
python textrank.py -m run -dn SemEval-2010 -fd title abstract full_text references
python textrank.py -m evaluate -dn SemEval-2010 -fd title abstract full_text references -en 3 5 7 10
:: TAICFpLp
python textrank.py -m run -dn SemEval-2010 -fd title abstract introduction conclusion body1 body2
python textrank.py -m evaluate -dn SemEval-2010 -fd title abstract introduction conclusion body1 body2 -en 3 5 7 10
:: TAICFpLpR
python textrank.py -m run -dn SemEval-2010 -fd title abstract introduction conclusion body1 body2 references
python textrank.py -m evaluate -dn SemEval-2010 -fd title abstract introduction conclusion body1 body2 references -en 3 5 7 10

:: PubMed
:: TA
python textrank.py -m run -dn PubMed -fd title abstract
python textrank.py -m evaluate -dn PubMed -fd title abstract -en 3 5 7 10
:: TAI
python textrank.py -m run -dn PubMed -fd title abstract introduction
python textrank.py -m evaluate -dn PubMed -fd title abstract introduction -en 3 5 7 10
:: TAC
python textrank.py -m run -dn PubMed -fd title abstract conclusion
python textrank.py -m evaluate -dn PubMed -fd title abstract conclusion -en 3 5 7 10
:: TAFp
python textrank.py -m run -dn PubMed -fd title abstract body1
python textrank.py -m evaluate -dn PubMed -fd title abstract body1 -en 3 5 7 10
:: TALp
python textrank.py -m run -dn PubMed -fd title abstract body2
python textrank.py -m evaluate -dn PubMed -fd title abstract body2 -en 3 5 7 10
:: TAR
python textrank.py -m run -dn PubMed -fd title abstract references
python textrank.py -m evaluate -dn PubMed -fd title abstract references -en 3 5 7 10
:: TAF
python textrank.py -m run -dn PubMed -fd title abstract full_text
python textrank.py -m evaluate -dn PubMed -fd title abstract full_text -en 3 5 7 10
:: TAFR
python textrank.py -m run -dn PubMed -fd title abstract full_text references
python textrank.py -m evaluate -dn PubMed -fd title abstract full_text references -en 3 5 7 10
:: TAICFpLp
python textrank.py -m run -dn PubMed -fd title abstract introduction conclusion body1 body2
python textrank.py -m evaluate -dn PubMed -fd title abstract introduction conclusion body1 body2 -en 3 5 7 10
:: TAICFpLpR
python textrank.py -m run -dn PubMed -fd title abstract introduction conclusion body1 body2 references
python textrank.py -m evaluate -dn PubMed -fd title abstract introduction conclusion body1 body2 references -en 3 5 7 10

:: LIS-2000
:: TA
python textrank.py -m run -dn LIS-2000 -fd title abstract
python textrank.py -m evaluate -dn LIS-2000 -fd title abstract -en 3 5 7 10
:: TAI
python textrank.py -m run -dn LIS-2000 -fd title abstract introduction
python textrank.py -m evaluate -dn LIS-2000 -fd title abstract introduction -en 3 5 7 10
:: TAC
python textrank.py -m run -dn LIS-2000 -fd title abstract conclusion
python textrank.py -m evaluate -dn LIS-2000 -fd title abstract conclusion -en 3 5 7 10
:: TAFp
python textrank.py -m run -dn LIS-2000 -fd title abstract body1
python textrank.py -m evaluate -dn LIS-2000 -fd title abstract body1 -en 3 5 7 10
:: TALp
python textrank.py -m run -dn LIS-2000 -fd title abstract body2
python textrank.py -m evaluate -dn LIS-2000 -fd title abstract body2 -en 3 5 7 10
:: TAR
python textrank.py -m run -dn LIS-2000 -fd title abstract references
python textrank.py -m evaluate -dn LIS-2000 -fd title abstract references -en 3 5 7 10
:: TAF
python textrank.py -m run -dn LIS-2000 -fd title abstract full_text
python textrank.py -m evaluate -dn LIS-2000 -fd title abstract full_text -en 3 5 7 10
:: TAFR
python textrank.py -m run -dn LIS-2000 -fd title abstract full_text references
python textrank.py -m evaluate -dn LIS-2000 -fd title abstract full_text references -en 3 5 7 10
:: TAICFpLp
python textrank.py -m run -dn LIS-2000 -fd title abstract introduction conclusion body1 body2
python textrank.py -m evaluate -dn LIS-2000 -fd title abstract introduction conclusion body1 body2 -en 3 5 7 10
:: TAICFpLpR
python textrank.py -m run -dn LIS-2000 -fd title abstract introduction conclusion body1 body2 references
python textrank.py -m evaluate -dn LIS-2000 -fd title abstract introduction conclusion body1 body2 references -en 3 5 7 10

:: 3. naivebayes
:: SemEval-2010
:: TA
python naivebayes.py -m train -dn SemEval-2010 -fd title abstract
python naivebayes.py -m test -dn SemEval-2010 -fd title abstract -en 3 5 7 10
:: TAI
python naivebayes.py -m train -dn SemEval-2010 -fd title abstract introduction
python naivebayes.py -m test -dn SemEval-2010 -fd title abstract introduction -en 3 5 7 10
:: TAC
python naivebayes.py -m train -dn SemEval-2010 -fd title abstract conclusion
python naivebayes.py -m test -dn SemEval-2010 -fd title abstract conclusion -en 3 5 7 10
:: TAFp
python naivebayes.py -m train -dn SemEval-2010 -fd title abstract body1
python naivebayes.py -m test -dn SemEval-2010 -fd title abstract body1 -en 3 5 7 10
:: TALp
python naivebayes.py -m train -dn SemEval-2010 -fd title abstract body2
python naivebayes.py -m test -dn SemEval-2010 -fd title abstract body2 -en 3 5 7 10
:: TAR
python naivebayes.py -m train -dn SemEval-2010 -fd title abstract references
python naivebayes.py -m test -dn SemEval-2010 -fd title abstract references -en 3 5 7 10
:: TAF
python naivebayes.py -m train -dn SemEval-2010 -fd title abstract full_text
python naivebayes.py -m test -dn SemEval-2010 -fd title abstract full_text -en 3 5 7 10
:: TAFR
python naivebayes.py -m train -dn SemEval-2010 -fd title abstract full_text references
python naivebayes.py -m test -dn SemEval-2010 -fd title abstract full_text references -en 3 5 7 10
:: TAICFpLp
python naivebayes.py -m train -dn SemEval-2010 -fd title abstract introduction conclusion body1 body2
python naivebayes.py -m test -dn SemEval-2010 -fd title abstract introduction conclusion body1 body2 -en 3 5 7 10
:: TAICFpLpR
python naivebayes.py -m train -dn SemEval-2010 -fd title abstract introduction conclusion body1 body2 references
python naivebayes.py -m test -dn SemEval-2010 -fd title abstract introduction conclusion body1 body2 references -en 3 5 7 10

:: PubMed
:: TA
python naivebayes.py -m train -dn PubMed -fd title abstract
python naivebayes.py -m test -dn PubMed -fd title abstract -en 3 5 7 10
:: TAI
python naivebayes.py -m train -dn PubMed -fd title abstract introduction
python naivebayes.py -m test -dn PubMed -fd title abstract introduction -en 3 5 7 10
:: TAC
python naivebayes.py -m train -dn PubMed -fd title abstract conclusion
python naivebayes.py -m test -dn PubMed -fd title abstract conclusion -en 3 5 7 10
:: TAFp
python naivebayes.py -m train -dn PubMed -fd title abstract body1
python naivebayes.py -m test -dn PubMed -fd title abstract body1 -en 3 5 7 10
:: TALp
python naivebayes.py -m train -dn PubMed -fd title abstract body2
python naivebayes.py -m test -dn PubMed -fd title abstract body2 -en 3 5 7 10
:: TAR
python naivebayes.py -m train -dn PubMed -fd title abstract references
python naivebayes.py -m test -dn PubMed -fd title abstract references -en 3 5 7 10
:: TAF
python naivebayes.py -m train -dn PubMed -fd title abstract full_text
python naivebayes.py -m test -dn PubMed -fd title abstract full_text -en 3 5 7 10
:: TAFR
python naivebayes.py -m train -dn PubMed -fd title abstract full_text references
python naivebayes.py -m test -dn PubMed -fd title abstract full_text references -en 3 5 7 10
:: TAICFpLp
python naivebayes.py -m train -dn PubMed -fd title abstract introduction conclusion body1 body2
python naivebayes.py -m test -dn PubMed -fd title abstract introduction conclusion body1 body2 -en 3 5 7 10
:: TAICFpLpR
python naivebayes.py -m train -dn PubMed -fd title abstract introduction conclusion body1 body2 references
python naivebayes.py -m test -dn PubMed -fd title abstract introduction conclusion body1 body2 references -en 3 5 7 10

:: LIS-2000
:: TA
python naivebayes.py -m train -dn LIS-2000 -fd title abstract
python naivebayes.py -m test -dn LIS-2000 -fd title abstract -en 3 5 7 10
:: TAI
python naivebayes.py -m train -dn LIS-2000 -fd title abstract introduction
python naivebayes.py -m test -dn LIS-2000 -fd title abstract introduction -en 3 5 7 10
:: TAC
python naivebayes.py -m train -dn LIS-2000 -fd title abstract conclusion
python naivebayes.py -m test -dn LIS-2000 -fd title abstract conclusion -en 3 5 7 10
:: TAFp
python naivebayes.py -m train -dn LIS-2000 -fd title abstract body1
python naivebayes.py -m test -dn LIS-2000 -fd title abstract body1 -en 3 5 7 10
:: TALp
python naivebayes.py -m train -dn LIS-2000 -fd title abstract body2
python naivebayes.py -m test -dn LIS-2000 -fd title abstract body2 -en 3 5 7 10
:: TAR
python naivebayes.py -m train -dn LIS-2000 -fd title abstract references
python naivebayes.py -m test -dn LIS-2000 -fd title abstract references -en 3 5 7 10
:: TAF
python naivebayes.py -m train -dn LIS-2000 -fd title abstract full_text
python naivebayes.py -m test -dn LIS-2000 -fd title abstract full_text -en 3 5 7 10
:: TAFR
python naivebayes.py -m train -dn LIS-2000 -fd title abstract full_text references
python naivebayes.py -m test -dn LIS-2000 -fd title abstract full_text references -en 3 5 7 10
:: TAICFpLp
python naivebayes.py -m train -dn LIS-2000 -fd title abstract introduction conclusion body1 body2
python naivebayes.py -m test -dn LIS-2000 -fd title abstract introduction conclusion body1 body2 -en 3 5 7 10
:: TAICFpLpR
python naivebayes.py -m train -dn LIS-2000 -fd title abstract introduction conclusion body1 body2 references
python naivebayes.py -m test -dn LIS-2000 -fd title abstract introduction conclusion body1 body2 references -en 3 5 7 10

:: 4. Crf
:: SemEval-2010
:: POS IWOT TI TR
:: TA
python crf.py -m train -dn SemEval-2010 -fd title abstract -fs POS IWOT TI TR
python crf.py -m test -dn SemEval-2010 -fd title abstract -fs POS IWOT TI TR
:: TAI
python crf.py -m train -dn SemEval-2010 -fd title abstract introduction -fs POS IWOT TI TR
python crf.py -m test -dn SemEval-2010 -fd title abstract introduction -fs POS IWOT TI TR
:: TAC
python crf.py -m train -dn SemEval-2010 -fd title abstract conclusion -fs POS IWOT TI TR
python crf.py -m test -dn SemEval-2010 -fd title abstract conclusion -fs POS IWOT TI TR
:: TAFp
python crf.py -m train -dn SemEval-2010 -fd title abstract body1 -fs POS IWOT TI TR
python crf.py -m test -dn SemEval-2010 -fd title abstract body1 -fs POS IWOT TI TR
:: TALp
python crf.py -m train -dn SemEval-2010 -fd title abstract body2 -fs POS IWOT TI TR
python crf.py -m test -dn SemEval-2010 -fd title abstract body2 -fs POS IWOT TI TR
:: TAR
python crf.py -m train -dn SemEval-2010 -fd title abstract references -fs POS IWOT TI TR
python crf.py -m test -dn SemEval-2010 -fd title abstract references -fs POS IWOT TI TR
:: TAF
python crf.py -m train -dn SemEval-2010 -fd title abstract full_text -fs POS IWOT TI TR
python crf.py -m test -dn SemEval-2010 -fd title abstract full_text -fs POS IWOT TI TR
:: TAFR
python crf.py -m train -dn SemEval-2010 -fd title abstract full_text references -fs POS IWOT TI TR
python crf.py -m test -dn SemEval-2010 -fd title abstract full_text references -fs POS IWOT TI TR
:: TAICFpLp
python crf.py -m train -dn SemEval-2010 -fd title abstract introduction conclusion body1 body2 -fs POS IWOT TI TR
python crf.py -m test -dn SemEval-2010 -fd title abstract introduction conclusion body1 body2 -fs POS IWOT TI TR
:: TAICFpLpR
python crf.py -m train -dn SemEval-2010 -fd title abstract introduction conclusion body1 body2 references -fs POS IWOT TI TR
python crf.py -m test -dn SemEval-2010 -fd title abstract introduction conclusion body1 body2 references -fs POS IWOT TI TR

:: PubMed
:: TA POS LEN WFOF WFF WFR IWOT IWOR TI TR
:: TA
python crf.py -m train -dn PubMed -fd title abstract -fs POS LEN WFOF WFF WFR IWOT IWOR TI TR
python crf.py -m test -dn PubMed -fd title abstract -fs POS LEN WFOF WFF WFR IWOT IWOR TI TR
:: TAI
python crf.py -m train -dn PubMed -fd title abstract introduction -fs POS LEN WFOF WFF WFR IWOT IWOR TI TR
python crf.py -m test -dn PubMed -fd title abstract introduction -fs POS LEN WFOF WFF WFR IWOT IWOR TI TR
:: TAC
python crf.py -m train -dn PubMed -fd title abstract conclusion -fs POS LEN WFOF WFF WFR IWOT IWOR TI TR
python crf.py -m test -dn PubMed -fd title abstract conclusion -fs POS LEN WFOF WFF WFR IWOT IWOR TI TR
:: TAFp
python crf.py -m train -dn PubMed -fd title abstract body1 -fs POS LEN WFOF WFF WFR IWOT IWOR TI TR
python crf.py -m test -dn PubMed -fd title abstract body1 -fs POS LEN WFOF WFF WFR IWOT IWOR TI TR
:: TALp
python crf.py -m train -dn PubMed -fd title abstract body2 -fs POS LEN WFOF WFF WFR IWOT IWOR TI TR
python crf.py -m test -dn PubMed -fd title abstract body2 -fs POS LEN WFOF WFF WFR IWOT IWOR TI TR
:: TAR
python crf.py -m train -dn PubMed -fd title abstract references -fs POS LEN WFOF WFF WFR IWOT IWOR TI TR
python crf.py -m test -dn PubMed -fd title abstract references -fs POS LEN WFOF WFF WFR IWOT IWOR TI TR
:: TAF
python crf.py -m train -dn PubMed -fd title abstract full_text -fs POS LEN WFOF WFF WFR IWOT IWOR TI TR
python crf.py -m test -dn PubMed -fd title abstract full_text -fs POS LEN WFOF WFF WFR IWOT IWOR TI TR
:: TAFR
python crf.py -m train -dn PubMed -fd title abstract full_text references -fs POS LEN WFOF WFF WFR IWOT IWOR TI TR
python crf.py -m test -dn PubMed -fd title abstract full_text references -fs POS LEN WFOF WFF WFR IWOT IWOR TI TR
:: TAICFpLp
python crf.py -m train -dn PubMed -fd title abstract introduction conclusion body1 body2 -fs POS LEN WFOF WFF WFR IWOT IWOR TI TR
python crf.py -m test -dn PubMed -fd title abstract introduction conclusion body1 body2 -fs POS LEN WFOF WFF WFR IWOT IWOR TI TR
:: TAICFpLpR
python crf.py -m train -dn PubMed -fd title abstract introduction conclusion body1 body2 references -fs POS LEN WFOF WFF WFR IWOT IWOR TI TR
python crf.py -m test -dn PubMed -fd title abstract introduction conclusion body1 body2 references -fs POS LEN WFOF WFF WFR IWOT IWOR TI TR

:: LIS-2000
:: POS WFOF WFR IWOT TI TR
:: TA
python crf.py -m train -dn LIS-2000 -fd title abstract -fs POS WFOF WFR IWOT TI TR
python crf.py -m test -dn LIS-2000 -fd title abstract -fs POS WFOF WFR IWOT TI TR
:: TAI
python crf.py -m train -dn LIS-2000 -fd title abstract introduction -fs POS WFOF WFR IWOT TI TR
python crf.py -m test -dn LIS-2000 -fd title abstract introduction -fs POS WFOF WFR IWOT TI TR
:: TAC
python crf.py -m train -dn LIS-2000 -fd title abstract conclusion -fs POS WFOF WFR IWOT TI TR
python crf.py -m test -dn LIS-2000 -fd title abstract conclusion -fs POS WFOF WFR IWOT TI TR
:: TAFp
python crf.py -m train -dn LIS-2000 -fd title abstract body1 -fs POS WFOF WFR IWOT TI TR
python crf.py -m test -dn LIS-2000 -fd title abstract body1 -fs POS WFOF WFR IWOT TI TR
:: TALp
python crf.py -m train -dn LIS-2000 -fd title abstract body2 -fs POS WFOF WFR IWOT TI TR
python crf.py -m test -dn LIS-2000 -fd title abstract body2 -fs POS WFOF WFR IWOT TI TR
:: TAR
python crf.py -m train -dn LIS-2000 -fd title abstract references -fs POS WFOF WFR IWOT TI TR
python crf.py -m test -dn LIS-2000 -fd title abstract references -fs POS WFOF WFR IWOT TI TR
:: TAF
python crf.py -m train -dn LIS-2000 -fd title abstract full_text -fs POS WFOF WFR IWOT TI TR
python crf.py -m test -dn LIS-2000 -fd title abstract full_text -fs POS WFOF WFR IWOT TI TR
:: TAFR
python crf.py -m train -dn LIS-2000 -fd title abstract full_text references -fs POS WFOF WFR IWOT TI TR
python crf.py -m test -dn LIS-2000 -fd title abstract full_text references -fs POS WFOF WFR IWOT TI TR
:: TAICFpLp
python crf.py -m train -dn LIS-2000 -fd title abstract introduction conclusion body1 body2 -fs POS WFOF WFR IWOT TI TR
python crf.py -m test -dn LIS-2000 -fd title abstract introduction conclusion body1 body2 -fs POS WFOF WFR IWOT TI TR
:: TAICFpLpR
python crf.py -m train -dn LIS-2000 -fd title abstract introduction conclusion body1 body2 references -fs POS WFOF WFR IWOT TI TR
python crf.py -m test -dn LIS-2000 -fd title abstract introduction conclusion body1 body2 references -fs POS WFOF WFR IWOT TI TR
