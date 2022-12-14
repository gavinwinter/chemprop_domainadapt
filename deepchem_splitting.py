import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import random

from deepchem.splits import FingerprintSplitter, ButinaSplitter, MolecularWeightSplitter, MaxMinSplitter
from deepchem.data import NumpyDataset

import matplotlib.pyplot as plt

df = pd.read_csv('../content/drive/MyDrive/6.7900_project/250k_rndm_zinc_drugs_clean_3.csv')
print(df)

# remove 
idxs = []
for index, row in tqdm(df.iterrows()):
  try:
    m = Chem.MolFromSmiles(df['smiles'][index])
    if m is not None:
      continue
  except:
    idxs.append(index)
    print(df[index]['smiles'])

df.drop(labels=idxs, axis=0)
print(df)

df = df.iloc[random.sample(range(0,240000), 30000)]
print(df)



ds = NumpyDataset.from_dataframe(df, X='smiles', y='logP', ids='smiles')
train, test = FingerprintSplitter().train_test_split(dataset=ds, frac_train=0.33333)
train, test = ButinaSplitter(cutoff=0.01).train_test_split(dataset=ds, frac_train=0.5)
train, test = MolecularWeightSplitter().train_test_split(dataset=ds, frac_train=0.5)
train, test = MaxMinSplitter().train_test_split(dataset=ds, frac_train=0.5)

source = train.to_dataframe()
target = test.to_dataframe()



fig, ax = plt.subplots()
ax.hist(source['y'], density=True, label='source', alpha=0.5)
ax.hist(target['y'], density=True, label='target', alpha=0.5) 
ax.legend()
ax.set_xlabel('logP')



source.drop(labels=['ids'], axis=1, inplace=True)
source.rename(columns = {'X':'smiles'}, inplace = True)
source.rename(columns = {'y':'logp'}, inplace = True)
source['label'] = 0

target.drop(labels=['ids'], axis=1, inplace=True)
target.rename(columns = {'X':'smiles'}, inplace = True)
target.rename(columns = {'y':'logp'}, inplace = True)
target['label'] = 1
print(source)
print(target)

classification = pd.concat([source,target], axis=0)
classification.to_csv('classification.csv', index=False)
source.to_csv('source.csv', index=False)

target = target.sample(frac=1)
target_holdout = target[:10000]
target_unlabeled = target[10000:]
target_holdout.to_csv('target_holdout.csv', index=False)
target_unlabeled.to_csv('target_unlabeled.csv', index=False)

