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
    print(
        'NAIVE - {mode}: Percentage of effective compounds = {unbiased_ratio}'
    )
    print(
        'BIASED - {mode}: Percentage of effective compounds = {biased_ratio}'
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
    s = site.replace('_', ' ')
    c = cell_line.replace('_', ' ')
    t2 = f'generator for {s} cancer. (cell: {c})'
    plt.title(t1 + t2, size=13)
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
            f'results/{mode}_{cell_line}_epoch_{epoch}_eff_{biased_ratio}.pdf'
        )
    )
    plt.clf()


def plot_loss(
    loss, reward, epoch, cell_line, save_path, rolling=1, site='unknown'
):
    loss = pd.Series(loss).rolling(rolling).mean()
    rewards = pd.Series(reward).rolling(rolling).mean()

    plt.plot(np.arange(len(loss)), loss, color='r')
    plt.ylabel('RL-loss (log softmax)', size=12).set_color('r')
    plt.xlabel('Training steps', size=12)
    # Plot KLD on second y axis
    _ = plt.twinx()
    s = site.replace('_', ' ')
    _ = cell_line.replace('_', ' ')
    plt.plot(np.arange(len(rewards)), rewards, color='g')
    plt.ylabel('Achieved rewards', size=12).set_color('g')
    plt.title('PaccMann$^{\mathrm{RL}}$ generator for ' + s + ' cancer')
    plt.savefig(
        os.path.join(save_path, f'results/loss_ep_{epoch}_cell_{cell_line}')
    )
    plt.clf()
