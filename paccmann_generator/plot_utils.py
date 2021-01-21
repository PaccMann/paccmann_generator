"""Plotting utilities."""
import os
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_and_compare(
    unbiased_preds, biased_preds, site, cell_line, epoch, save_path, mode, bs
):
    biased_ratio = np.round(
        100 * (np.sum(biased_preds < 0) / len(biased_preds)), 1
    )
    unbiased_ratio = np.round(
        100 * (np.sum(unbiased_preds < 0) / len(unbiased_preds)), 1
    )
    print(f'Site: {site}, cell line: {cell_line}')
    print(
        f'NAIVE - {mode}: Percentage of effective compounds = {unbiased_ratio}'
    )
    print(
        f'BIASED - {mode}: Percentage of effective compounds = {biased_ratio}'
    )

    fig, ax = plt.subplots()
    sns.kdeplot(
        unbiased_preds,
        shade=True,
        color='grey',
        label=f'Unbiased: {unbiased_ratio}% '
    )
    sns.kdeplot(
        biased_preds,
        shade=True,
        color='red',
        label=f'Optimized: {biased_ratio}% '
    )
    valid = f'SMILES validity: \n {round((len(biased_preds)/bs) * 100, 1)}%'
    txt = "$\mathbf{Drug \ efficacy}$: "
    handles, labels = plt.gca().get_legend_handles_labels()
    patch = mpatches.Patch(color='none', label=txt)

    handles.insert(0, patch)  # add new patches and labels to list
    labels.insert(0, txt)

    plt.legend(handles, labels, loc='upper right')
    plt.xlabel('Predicted log(micromolar IC50)')
    plt.ylabel(f'Density of generated molecules (n={bs})')
    t1 = 'PaccMann$^{\mathrm{RL}}$ '
    #s = site.replace('_', ' ')
    #c = cell_line.replace('_', ' ')
    #t2 = f'generator for {s} cancer. (cell: {c})'
    plt.title(t1 , size=13) #+ t2
    plt.text(0.67, 0.70, valid, weight='bold', transform=plt.gca().transAxes)
    plt.text(
        0.05,
        0.8,
        'Effective compounds',
        weight='bold',
        color='grey',
        transform=plt.gca().transAxes
    )
    ax.axvspan(-10, 0, alpha=0.5, color=[0.85, 0.85, 0.85])
    plt.xlim([-4, 8])
    plt.savefig(
        os.path.join(
            save_path,
            f'results/{mode}_epoch_{epoch}_eff_{biased_ratio}.pdf'
        )
    )
    plt.clf()


def plot_and_compare_proteins(
    unbiased_preds, biased_preds, protein, epoch, save_path, mode, bs
):

    biased_ratio = np.round(
        100 * (np.sum(biased_preds > 0.5) / len(biased_preds)), 1
    )
    unbiased_ratio = np.round(
        100 * (np.sum(unbiased_preds > 0.5) / len(unbiased_preds)), 1
    )
    print(
        f'NAIVE - {mode}: Percentage of binding compounds = {unbiased_ratio}'
    )
    print(f'BIASED - {mode}: Percentage of binding compounds = {biased_ratio}')

    fig, ax = plt.subplots()
    sns.distplot(
        unbiased_preds,
        kde_kws={
            'shade': True,
            'alpha': 0.5,
            'linewidth': 2,
            'clip': [0, 1],
            'kernel': 'cos'
        },
        color='grey',
        label=f'Unbiased: {unbiased_ratio}% ',
        kde=True,
        rug=True,
        hist=False
    )
    sns.distplot(
        biased_preds,
        kde_kws={
            'shade': True,
            'alpha': 0.5,
            'linewidth': 2,
            'clip': [0, 1],
            'kernel': 'cos'
        },
        color='red',
        label=f'Optimized: {biased_ratio}% ',
        kde=True,
        rug=True,
        hist=False
    )
    valid = f'SMILES validity: {round((len(biased_preds)/bs) * 100, 1)}%'
    txt = "$\mathbf{Drug \ binding}$: "
    handles, labels = plt.gca().get_legend_handles_labels()
    patch = mpatches.Patch(color='none', label=txt)

    handles.insert(0, patch)  # add new patches and labels to list
    labels.insert(0, txt)

    plt.legend(handles, labels, loc='upper left')
    plt.xlabel('Predicted binding probability')
    plt.ylabel(f'Density of generated molecules')
    t1 = 'PaccMann$^{\mathrm{RL}}$ '
    # protein_name = '_'.join(protein.split('=')[1].split('-')[:-1])
    # organism = protein.split('=')[-1]
    # t2 = f'generator for: {protein_name}\n({organism})'
    protein_name = protein
    #t2 = f'generator for: {protein_name}\n'
    plt.title(t1, size=10) # + t2
    plt.text(
        0.55,
        0.95,
        'Predicted as binding',
        weight='bold',
        color='grey',
        transform=plt.gca().transAxes
    )
    ax.axvspan(0.5, 1.2, alpha=0.5, color=[0.85, 0.85, 0.85])
    plt.xlim([0., 1.])
    plt.savefig(
        os.path.join(
            save_path,
            f'results/{mode}_epoch_{epoch}_aff_{biased_ratio}.pdf'
        )
    )
    plt.clf()

def plot_loss(
    loss, reward, epoch, save_path, rolling=1, site='unknown'
):
    idx = np.argmax(reward)
    loss = pd.Series(loss).rolling(rolling).mean()
    rewards = pd.Series(reward).rolling(rolling).mean()
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(loss)), loss, color='r')
    ax.set_ylabel('RL-loss (log softmax)', size=12).set_color('r')
    ax.set_xlabel('Training epochs', size=12)  
    ax.set_xticklabels(np.arange(0, 61, 10))
    minor_ticks = np.arange(0, 301, 5)
    ax.set_xticks(minor_ticks, minor=True)
    majnor_ticks = np.arange(0, 301, 50)
    ax.set_xticks(majnor_ticks)
    ax.grid(b=True,axis='x', which='both', linestyle='-')
    ax.grid(which='minor',axis='x', alpha=0.2)
    ax.grid(which='major',axis='x', alpha=0.5)
    # Plot KLD on second y axis
    ax2 = ax.twinx()
    s = site.replace('_', ' ')
    #_ = cell_line.replace('_', ' ')
    ax2.plot(np.arange(len(rewards)), rewards, color='g')
    ax2.grid(True, axis='x', which='both')
    ax2.ylabel('Achieved rewards', size=12).set_color('g')
    ax2.title('PaccMann$^{\mathrm{RL}}$ \max reward at epoch:'+ str(int(idx/5)+1))
    fig.savefig(
        os.path.join(save_path, f'results/loss_ep_{epoch}')
    )
    print(os.path.join(save_path, f'results/loss_ep_{epoch}'))
    plt.clf()