import pandas as pd
import numpy as np
model='liver_concat_allValid_SNU-423_combined'
mols = pd.read_csv('biased_models/'+model+'/results/generated.csv', index_col=0)
loss = pd.read_csv('biased_models/'+model+'/results/loss_reward_evolution.csv', index_col=0)
max_epoch = np.max(loss['epoch'])
assert(max_epoch == np.max(mols['epoch']))
loss_tot = loss[loss['epoch']==max_epoch]
print(loss_tot.shape, loss_tot)
idx2 = 0
for idx in range(1,max_epoch):
    loss_current_epoch = loss[loss['epoch']==idx].tail(5)
    print(idx2,idx2+loss_current_epoch.shape[0])
    loss_tot.loc[idx2:idx2+4, :] = loss_current_epoch
    idx2=idx2+5
print(loss_tot)
loss_tot.to_csv('biased_models/'+model+'/results/loss_and_more.csv')
    #1/0
