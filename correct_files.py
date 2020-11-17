import pandas as pd
import numpy as np
models='liver_concat_sanitized_SNU-423_'
end = ['combined', 'omics', 'protein']

def main(model):
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
    mols_tot = mols[mols['epoch']==max_epoch]
    old_epoch=0
    previous_epoch=pd.DataFrame(columns= mols.columns)
    for idx in range(1,max_epoch+1):
        #print("start",mols[mols['epoch']==idx].shape)
        temp = mols[mols['epoch']==idx]
        current_epoch = temp[old_epoch:]
        #print(idx, current_epoch.shape)
        mols_tot.loc[old_epoch:old_epoch+current_epoch.shape[0]-1,:] = current_epoch
        #print(old_epoch, old_epoch + current_epoch.shape[0])
        old_epoch = old_epoch+current_epoch.shape[0]
        previous_epoch = temp
    mols_tot.to_csv('biased_models/'+model+'/results/generated_per_epoch.csv')

for e in end:
    model = models + e
    main(model)