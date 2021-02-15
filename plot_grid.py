import pandas as pd
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interact, interactive, fixed, interact_manual
import numpy as np

def plotting_model(data, file_path, name):
    fig, axs = plt.subplots(7, 3, figsize=(20, 30))
    for ii, i in enumerate(data.index):
        if (i not in ['x', 'y']):
            print(i)
            df = data.loc[[i,'x','y']]
            sc = axs.flat[ii].scatter(x=df.loc['x'], y = df.loc['y'], c = df.loc[i])
            fig.colorbar(sc, ax=axs.flat[ii])
            axs.flat[ii].set_title(i)
    plt.suptitle(name, fontsize=16)
    plt.show()
    #fig.savefig(file_path + "grid_metrics.pdf", transparent=False)


dict_models = {}
for model in ['average', 'concat']:
    for part in ['omics','protein']:
        print(model, part)
        file_name = 'metrics_small_std.csv'
        file_path = 'biased_models/liver_' + model + '_sanitized_SNU-423_' + part + '/results/'
        file_path_coo = 'biased_models/liver_' + model + '_sanitized_SNU-423_' + part + '/results/grid_coordinates.csv'
        data = pd.read_csv(file_path + file_name, index_col = 0)
        coordinates = pd.read_csv(file_path_coo, header = None, index_col = None).T
        data.index = ['valid','unique@1000','unique@10000','FCD/UGTest','SNN/UGTest','Frag/UGTest','Scaf/UGTest','FCD/UGTrain','SNN/UGTrain','Frag/UGTrain','Scaf/UGTrain','IntDiv','IntDiv2','Filters','logP','SA','QED','weight','Novelty']
        #print(data)
        data.loc['FCD/UGTest',:] =data.loc['FCD/UGTest',:]/100
        data.loc['FCD/UGTrain',:] = data.loc['FCD/UGTrain',:]/100
        #data = data.drop('weight')
        tmp = [i for i in coordinates.loc[0]]
        data.loc['x'] = tmp
        tmp = [i for i in coordinates.loc[1]]
        data.loc['y'] = tmp
        dict_models[model+'_'+part] = data
        plotting_model(data, file_path, model+'_'+part)



def plotting(dict_models):
    dict_metric = {}
    fig, axs = plt.subplots(6,3, figsize=(20, 30))
    for ii, i in enumerate(data.index):
        print(i)
        for model in ['average', 'concat']:
            for part in ['omics', 'protein']:
                dict_metric[model+'_'+part] = dict_models[model+'_'+part].loc[i]
        #print(dict_metric)
        metric = pd.DataFrame(dict_metric).T
        
        sns.heatmap(metric, ax=axs.flat[ii]) # , vmin=0, vmax=10
        axs.flat[ii].set_title(i)
    plt.show()
    fig.savefig("grid_metrics.pdf", transparent=False)
    