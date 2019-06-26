import urllib.request, json, ssl

#bypass SSL verification
context = ssl._create_unverified_context()

#you will need your own API. play around with key= parm also
with urllib.request.urlopen("https://manifesto-project.wzb.eu/tools/api_get_core.json?api_key=d00c54e1a64ef97f7a032c91ff45a627&key=MPDS2018b", context=context) as url:
        cmp_test = json.loads(url.read().decode())

#returns a list

#basic packages
import pandas as pd
import numpy as np

#create index col
index= list(range(len(cmp_test)))

#turn imported data into pd dataframe
d = pd.DataFrame(data = cmp_test,columns=cmp_test[0], index=index)

#checks:
len(d) #this year N=3925
#say we want 2/3 to be the training set:
#train_amount = round(len(d)*2/3)
#and the rest to be the validation set:
#val_amount = 1- train_amount

#slice data (non-random):
#train_set = d.iloc[[1,train_amount],:]
#val_set = d.iloc[[train_amount + 1, len(d) - 1],:]

#carry on with the actual training
