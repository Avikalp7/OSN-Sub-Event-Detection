import csv
import pickle
from datetime import datetime
d = {}
with open('hydb_acl.txt') as f:
    reader = csv.reader(f, delimiter="\t")
    # d = list(reader)
    temp_d = list(reader)
    for idx in xrange(0, len(temp_d)):
    	d[idx] = [temp_d[idx][3], datetime.strptime(temp_d[idx][0], '%Y-%m-%d %H:%M:%S')]
pickle.dump(d, open( "tweets_dict.p", "wb" ) )
