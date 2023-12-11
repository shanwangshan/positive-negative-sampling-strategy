import numpy as np
import random
import pandas as pd
from sklearn import preprocessing
vggsound =pd.read_csv( '../../vggsound-1.csv',header=None)

unwanted = pd.read_csv('../../unwanted.csv',header = None)
unwanted = unwanted[0].values.tolist()
vggsound = vggsound[~vggsound[0].isin(unwanted)]
train_files = vggsound.loc[vggsound[3] == 'train']
print('all together the number of training files is', len(train_files))
train_files.columns =['filename', 'st_time', 'label','type']
min_class = train_files['label'].value_counts().min()
classes = sorted(list(set(train_files['label'].values.tolist())))
indexes=[]
for i in classes:
    index_class = train_files.label[train_files.label.eq(i)].sample(n= min_class,random_state=1).index
    indexes.append(index_class.values.tolist())
selected_index = sum(indexes,[])
train_files = train_files.loc[selected_index]
le = preprocessing.LabelEncoder()
le.fit(train_files.label)
train_files['label'] = le.transform(train_files.label)




sudo_labels = []
for i in range(309):
    randomlist = random.sample(range(0, 123), int(123*0.3))
    for j in range(123):
        if j in randomlist:
            all_class =  list(np.arange(309))
            all_class.pop(i)
            sudo_labels.append(random.sample(all_class,1)[0])
        else:
            sudo_labels.append(i)
train_files['sudo_label'] = sudo_labels
train_files.to_csv('selected_files_sudo_labels.csv',sep='\t')
import pdb; pdb.set_trace()
