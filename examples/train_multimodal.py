import argparse
import json
import logging
import os
import sys
import numpy as np
import pandas as pd
import warnings
import torch
warnings.filterwarnings("ignore")
from paccmann_chemistry.models import (
    StackGRUDecoder, StackGRUEncoder, TeacherVAE
)
from paccmann_chemistry.utils import get_device
from paccmann_generator.plot_utils import plot_and_compare, plot_and_compare_proteins, plot_loss
from paccmann_generator.utils import add_avg_profile, omics_data_splitter, protein_data_splitter
from paccmann_omics.encoders import ENCODER_FACTORY
from pytoda.smiles.smiles_language import SMILESLanguage
from pytoda.proteins.protein_language import ProteinLanguage
from paccmann_generator.utils import disable_rdkit_logging
from paccmann_predictor.models import MODEL_FACTORY as MODEL_FACTORY_OMICS
from paccmann_sets.models.sets_autoencoder import SetsAE
import sys
sys.path.append('/home/tol/paccmann_affinity')
from paccmann_affinity.models.predictors import MODEL_FACTORY as MODEL_FACTORY_PROTEIN

from paccmann_generator.model import Model
from paccmann_generator.reinforce_proteins_omics import ReinforceProteinOmics
from paccmann_generator import ReinforceOmic
from paccmann_generator.reinforce_proteins import ReinforceProtein
from files import *
cancer_cell_lines = ['HUH-6-clone5','HuH-7','SNU-475','SNU-423','SNU-387','SNU-449','HLE','C3A']

# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('train_paccmann_rl')
logger_m = logging.getLogger('matplotlib')
logger_m.setLevel(logging.WARNING)

disable_rdkit_logging()

params = dict()
params_o, params_p = dict(), dict()
params['site'] = site
params_o['site'] = site
params_p['site'] = site
params['cancertype'] = cancertype

with open(params_path) as f:
    params.update(json.load(f))
with open(params_path) as f:
    params_o.update(json.load(f))
with open("examples/affinity/conditional_generator.json") as f:
    params_p.update(json.load(f))

# Load omics profiles for conditional generation,
# complement with avg per site
omics_df = pd.read_pickle(omics_data_path)
omics_df = add_avg_profile(omics_df)
idx = [i in cancer_cell_lines for i in omics_df['cell_line']]
omics_df  = omics_df[idx]
#omics_df = omics_df[omics_df.histology == cancertype]
print("omics data:", omics_df.shape, omics_df['cell_line'].iloc[0])
test_cell_line = omics_df['cell_line'].iloc[0]
print("test_cell_line:", test_cell_line)
model_name = model_name + '_'+ test_cell_line + '_lern' + str(params['learning_rate'])
logger.info(f'Model with name {model_name} starts.')

# Load protein sequence data
protein_df = pd.read_csv(protein_data_path, index_col=0)
protein_df = protein_df[~protein_df.index.isnull()]
protein_df.index = [i.split('|')[2] for i in protein_df.index]
protein_seq_df = pd.read_csv(protein_data_seq_path, names = ['sequence'], index_col=0)
protein_seq_df.index = [i.split('|')[2] for i in protein_seq_df.index]
protein_df = pd.concat([protein_df, protein_seq_df], axis=1, join='outer')
protein_df = protein_df[[s.split('_')[0] in cancer_genes for s in protein_df.index]]
print("proteins:", protein_df.index, len(cancer_genes))

# Specifies the baseline model used for comparison
unbiased_preds_df = pd.read_csv(unbiased_predictions_path)

model_types = ['onlyConcat', 'concat', 'average', 'set']
models = []
for m in model_types:
    model = Model(m, params, params_o, params_p, omics_df, protein_df, logger, 'test')
    models.append(model)

# Split the samples for conditional generation and initialize training
train_omics, test_omics = omics_data_splitter(
    omics_df, site, params.get('test_fraction', 0.2)
)
train_protein = protein_df
test_protein = protein_df
print("test omics:", len(test_omics), ":test protein", test_protein.shape, ":train omics:", len(train_omics), ":train protein:", train_protein.shape)

logger.info('Models restored, start training.')

# choose a validation cell line and protein.
eval_cell_lines = [test_cell_line]
eval_protein_names = np.random.choice(test_protein.index, size = 20, replace=False)
eval_protein_names = [str(i) for i in eval_protein_names]

for epoch in range(1, params['epochs'] + 1):
    for m in models:
        m.reset_metrics()
    
    for step in range(1, params['steps'] + 1):

        # Randomly sample a cell lines and proteins for training:
        cell_line  = [i for i in omics_df['cell_line'] if i != test_cell_line]
        protein_name = np.random.choice(train_protein.index, size = 20, replace=False)
        protein_name = [str(i) for i in protein_name]
        print(f'Current train cell_lines: {cell_line}')
        print(f'Current train proteins: {protein_name}')
        for m in models:
            m.train_and_save_steps(params['batch_size'], epoch, protein_name=protein_name, cell_line=cell_line)

    # Save model
    unbiased_predsP = unbiased_preds_df[protein_name].values.reshape(-1)[:params['batch_size']]
    unbiased_predsO = unbiased_preds_df[cell_line].values.reshape(-1)[:params['batch_size']]
    for m in models:
        m.model.save(f'gen_{epoch}.pt', f'enc_{epoch}_protein.pt', f'enc_{epoch}_omics.pt')
        m.save_loss_reward(epoch, params['epochs'])
        print(m.type)
        m.generate_and_save(epoch, 'train', params['batch_size'], unbiased_predsP=unbiased_predsP, protein_name=protein_name, 
            unbiased_predsO=unbiased_predsO, cell_line=cell_line)
    
    # Evaluate on a validation cell line and protein.
    unbiased_predsP = unbiased_preds_df[eval_protein_names].values.reshape(-1)[:params['eval_batch_size']]
    unbiased_predsO = unbiased_preds_df[eval_cell_lines].values.reshape(-1)[:params['eval_batch_size']]
    
    for m in models:
        print(m.type)
        m.generate_and_save(epoch, 'test', params['eval_batch_size'], unbiased_predsP=unbiased_predsP, protein_name=eval_protein_names, 
            unbiased_predsO=unbiased_predsO, cell_line=eval_cell_lines)