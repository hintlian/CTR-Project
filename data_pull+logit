import random as rd
import math
import sys
import string
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import *
import time

t0 = time.time()
src = 'train_aug.csv'
dst = 'dataSample.txt'

# This is the number of lines in our training data file 'train_rev2'
N = 40428967
rd.seed(1410)
# This is sample number that we want to draw from the training data
# You can change the next to set a sample size
if(len(sys.argv) > 1):
	sampleNum = int(sys.argv[1])
else:
	sampleNum = 5000000

shuffle = range(N+1)
sample = range(sampleNum+1);
sample[0] = -1
for x in range(sampleNum):
	tmp = shuffle[x+1]
	index = rd.randint(x+1,N)
	shuffle[x+1] = shuffle[index]
	shuffle[index] = tmp
	sample[x+1] = shuffle[x+1];


sample.sort()

fileSrc = open(src)
fileDst = open(dst,'wt')
fileDst.truncate()

i = -1;
index = 0;
for line in fileSrc:
	if(i == sample[index]):
		#write this line to file
		fileDst.write(line)
	 	index = index + 1
	 	if(index >= sampleNum):
	 		break;
	i = i+1

fileSrc.close()
fileDst.close()

  
#d = DataFrame(rawdata)

from sklearn.feature_extraction import DictVectorizer

vec= DictVectorizer()

categorical_feature=[]

cat_pos = ['day', 'hour','C1', 'banner_pos', 'site_id', 'site_domain','site_category', 'app_id' , 'app_domain' , 'app_category','device_id','device_ip', 'device_model', 'device_type','device_conn_type','C14', 'C15','C16','C17','C18','C19','C20', 'C21']         
print "loading"
train = pd.read_csv('dataSample.txt')
test = pd.read_csv('test_aug.csv')
print "fixing"
#for line in train:

for feature in cat_pos:
        train[feature] = map(str, train[feature])
        test[feature] = map(str, test[feature])
        categorical_feature.append(feature)

train_sparse = vec.fit_transform(train[categorical_feature].T.to_dict().values())
test_sparse = vec.transform(test[categorical_feature].T.to_dict().values())

logreg = LogisticRegression(penalty = 'l2', dual = False ,tol = 0.0001, C = .165)
print "training"
logreg.fit(train_sparse,train['click'])

test['click'] = 0
print "fitting"
predictions = logreg.predict_proba(test_sparse)

test['click'] = predictions[:,1]
test.to_csv('subnew10.csv', cols = ('id','click'),index=False)

t1 = time.time()

print (t1-t0)
