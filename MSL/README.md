`process_msl.py` script used to prepare data for training. In keeping with previous work we only focus on three of the hardest non-trivial instances from this dataset: A-4, C-2, and T-1. Data is already bound [-1, 1] so no scaling is applied

data taken from:
https://s3-us-west-2.amazonaws.com/telemanom/data.zip

labeled anomalies from:
https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv

Corresponding paper:
Hundman, Kyle, Valentino Constantinou, Christopher Laporte, Ian Colwell, and Tom Soderstrom. "Detecting spacecraft anomalies using lstms and nonparametric dynamic thresholding." In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining, pp. 387-395. 2018.