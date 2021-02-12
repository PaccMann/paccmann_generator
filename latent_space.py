import argparse
import json
import logging
import os
import sys
import numpy as np
import pandas as pd
import warnings
import glob
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
import sys
import torch
#sys.path.append('/dataP/tol/paccmann_affinity')
sys.path.append('/home/tol/paccmann_affinity')
from paccmann_affinity.models.predictors import MODEL_FACTORY as MODEL_FACTORY_PROTEIN

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
params['site'] = site
params['cancertype'] = cancertype


logger.info(f'Model with name {model_name} starts.')

# Load omics profiles for conditional generation,
# complement with avg per site
omics_df = pd.read_pickle(omics_data_path)
omics_df = add_avg_profile(omics_df)
idx = [i in cancer_cell_lines for i in omics_df['cell_line']]
omics_df  = omics_df[idx]
print("omics data:", omics_df.shape, omics_df['cell_line'].iloc[0])
test_cell_line = omics_df['cell_line'].iloc[0]
model_name = model_name + '_' + test_cell_line
#omics_df = omics_df[omics_df.histology == cancertype]

# Load protein sequence data
#if protein_data_path.endswith('.smi'):
#    protein_df = read_smi(protein_data_path, names=['Sequence'])
#elif protein_data_path.endswith('.csv'):
#    protein_df = pd.read_csv(protein_data_path, index_col=0, header=None, names=[str(x) for x in range(768)]) #'entry_name')
#else:
#    raise TypeError(
#        f"{protein_data_path.split('.')[-1]} files are not supported."
#    )

protein_df = pd.read_csv(protein_data_path, index_col=0)#, header=None, names=[str(x) for x in range(768)]) #'entry_name')
protein_df = protein_df[~protein_df.index.isnull()]
protein_df.index = [i.split('|')[2] for i in protein_df.index]
protein_seq_df = pd.read_csv(protein_data_seq_path, names = ['sequence'], index_col=0) #, index_col='entry_name')
#print(protein_seq_df.head)
protein_seq_df.index = [i.split('|')[2] for i in protein_seq_df.index]
protein_df = pd.concat([protein_df, protein_seq_df], axis=1, join='outer')
protein_df = protein_df[[s.split('_')[0] in cancer_genes for s in protein_df.index]]
#print(protein_df.head)
print("proteins:", protein_df.index, len(cancer_genes))

with open(params_path) as f:
    params.update(json.load(f))

# Define network
with open(os.path.join(omics_model_path, 'model_params.json')) as f:
    cell_params = json.load(f)

cell_encoder = ENCODER_FACTORY['dense'](cell_params)
cell_encoder.load(
    os.path.join(
        omics_model_path,
        f"weights/best_{params.get('omics_metric','both')}_encoder.pt"
    ),
    map_location=get_device()
)
cell_encoder.eval()


# Restore SMILES Model
with open(os.path.join(mol_model_path, 'model_params.json')) as f:
    mol_params = json.load(f)

gru_encoder = StackGRUEncoder(mol_params)
gru_decoder = StackGRUDecoder(mol_params)
generator = TeacherVAE(gru_encoder, gru_decoder)
generator.load(
    os.path.join(
        mol_model_path,
        f"weights/best_{params.get('smiles_metric', 'rec')}.pt"
    ),
    map_location=get_device()
)
# Load languages
generator_smiles_language = SMILESLanguage.load(
    os.path.join(mol_model_path, 'selfies_language.pkl')
)
#generator._associate_language(generator_smiles_language)

 # Restore protein model
with open(os.path.join(protein_model_path, 'model_params.json')) as f:
    protein_params = json.load(f)

# Define network
protein_encoder = ENCODER_FACTORY['dense'](protein_params)
protein_encoder.load(
    os.path.join(
        protein_model_path,
        f"weights/best_{params.get('omics_metric','both')}_encoder.pt"
    ),
    map_location=get_device()
)
protein_encoder.eval()

# Restore omics model
with open(os.path.join(omics_model_path, 'model_params.json')) as f:
    cell_params = json.load(f)

# Define network
cell_encoder = ENCODER_FACTORY['dense'](cell_params)
cell_encoder.load(
    os.path.join(
        omics_model_path,
        f"weights/best_{params.get('omics_metric','both')}_encoder.pt"
    ),
    map_location=get_device()
)
cell_encoder.eval()

#load predictors
with open(os.path.join(ic50_model_path, 'model_params.json')) as f:
    paccmann_params = json.load(f)


paccmann_predictor = MODEL_FACTORY_OMICS['mca'](paccmann_params)
paccmann_predictor.load(
    os.path.join(
        ic50_model_path,
        f"weights/best_{params.get('ic50_metric', 'mse')}_mca.pt"
    ),
    map_location=get_device()
)
paccmann_predictor.eval()
paccmann_smiles_language = SMILESLanguage.load(
    os.path.join(ic50_model_path, 'smiles_language.pkl')
)
paccmann_predictor._associate_language(paccmann_smiles_language)

with open(os.path.join(affinity_model_path, 'model_params.json')) as f:
    protein_pred_params = json.load(f)

protein_predictor = MODEL_FACTORY_PROTEIN['bimodal_mca'](protein_pred_params)
protein_predictor.load(
    os.path.join(
        affinity_model_path,
        f"weights/best_{params.get('p_metric', 'ROC-AUC')}_bimodal_mca.pt"
    ),
    map_location=get_device()
)
protein_predictor.eval()

affinity_smiles_language = SMILESLanguage.load(
    os.path.join(affinity_model_path, 'smiles_language.pkl')
)
affinity_protein_language = ProteinLanguage.load(
    os.path.join(affinity_model_path, 'protein_language.pkl')
)
protein_predictor._associate_language(affinity_smiles_language)
protein_predictor._associate_language(affinity_protein_language)

# Specifies the baseline model used for comparison
unbiased_preds_df = pd.read_csv(unbiased_predictions_path)
# yapf: disable

remove_invalid = remove_invalid

# Specifies the baseline model used for comparison
baseline = ReinforceProteinOmics(generator, protein_encoder, cell_encoder, \
    protein_predictor, paccmann_predictor, protein_df, omics_df, \
    params, generator_smiles_language, 'baseline', logger, remove_invalid
)

#############################################
# Create a fresh model that will be optimized
gru_encoder_rl = StackGRUEncoder(mol_params)
gru_decoder_rl = StackGRUDecoder(mol_params)
generator_rl = TeacherVAE(gru_encoder_rl, gru_decoder_rl)
generator_rl.load(
    os.path.join(
        mol_model_path, f"weights/best_{params.get('metric', 'rec')}.pt"
    ),
    map_location=get_device()
)
generator_rl.eval()
#generator_rl._associate_language(generator_smiles_language)

cell_encoder_rl = ENCODER_FACTORY['dense'](cell_params)
cell_encoder_rl.load(
    os.path.join(
        omics_model_path,
        f"weights/best_{params.get('metric', 'both')}_encoder.pt"
    ),
    map_location=get_device()
)
cell_encoder_rl.eval()

protein_encoder_rl = ENCODER_FACTORY['dense'](protein_params)
protein_encoder_rl.load(
    os.path.join(
        protein_model_path,
        f"weights/best_{params.get('metric', 'both')}_encoder.pt"
    ),
    map_location=get_device()
)
protein_encoder_rl.eval()

model_folder_name = site + '_' + model_name + '_combined'
learner_combined = ReinforceProteinOmics(
    generator_rl, protein_encoder_rl,cell_encoder_rl, \
    protein_predictor, paccmann_predictor, protein_df, omics_df, \
    params, generator_smiles_language, model_folder_name, logger, remove_invalid
)

gru_encoder_rl_o = StackGRUEncoder(mol_params)
gru_decoder_rl_o = StackGRUDecoder(mol_params)
generator_rl_o = TeacherVAE(gru_encoder_rl_o, gru_decoder_rl_o)
generator_rl_o.load(
    os.path.join(
        mol_model_path, f"weights/best_{params.get('metric', 'rec')}.pt"
    ),
    map_location=get_device()
)
generator_rl_o.eval()
#generator_rl._associate_language(generator_smiles_language)

cell_encoder_rl_o = ENCODER_FACTORY['dense'](cell_params)
cell_encoder_rl_o.load(
    os.path.join(
        omics_model_path,
        f"weights/best_{params.get('metric', 'both')}_encoder.pt"
    ),
    map_location=get_device()
)
cell_encoder_rl_o.eval()

model_folder_name = site + '_' + model_name + '_omics'
learner_omics = ReinforceOmic(
    generator_rl_o, cell_encoder_rl_o, paccmann_predictor, omics_df, params,
    generator_smiles_language, model_folder_name, logger, remove_invalid
)

gru_encoder_rl_p = StackGRUEncoder(mol_params)
gru_decoder_rl_p = StackGRUDecoder(mol_params)
generator_rl_p = TeacherVAE(gru_encoder_rl_p, gru_decoder_rl_p)
generator_rl_p.load(
    os.path.join(
        mol_model_path, f"weights/best_{params.get('metric', 'rec')}.pt"
    ),
    map_location=get_device()
)
generator_rl_p.eval()
#generator_rl._associate_language(generator_smiles_language)

protein_encoder_rl_p = ENCODER_FACTORY['dense'](protein_params)
protein_encoder_rl_p.load(
    os.path.join(
        protein_model_path,
        f"weights/best_{params.get('metric', 'both')}_encoder.pt"
    ),
    map_location=get_device()
)
protein_encoder_rl_p.eval()

model_folder_name = site + '_' + model_name + '_protein'
learner_protein = ReinforceProtein(
    generator_rl_p, protein_encoder_rl_p, protein_predictor, protein_df, params,
    generator_smiles_language, model_folder_name, logger, remove_invalid
)
models = [learner_combined, learner_omics, learner_protein]
print_stuff=False

#get latent space of the omics data:
latent = []
for gene in omics_df.cell_line:
    latent.append(learner_omics.encode_cell_line(cell_line=[gene], batch_size=1).numpy().tolist()[0][0])
assert(len(latent)==omics_df.shape[0])
omics_df = omics_df.join(pd.DataFrame(
    latent, 
    index=omics_df.index, 
    columns=range(128)
))
#omics_df['latent']=latent
omics_df.to_csv('/home/tol/data/gsdc_omics_latent.csv')

#get latent space of the protein data:
latent = []
for prot in protein_df.index:
    latent.append(learner_protein.encode_protein(protein=[prot], batch_size=1).numpy().tolist()[0][0])
assert(len(latent)==protein_df.shape[0])
protein_df = protein_df.join(pd.DataFrame(
    latent, 
    index=protein_df.index, 
    columns=range(128)
))
#protein_df['latent']=latent
protein_df.to_csv('/home/tol/data/gsdc_proteins_latent.csv')
1/0
# get altent space of the generated smiles
for model in models:
    data = pd.read_csv(glob.glob(os.path.join(model.model_path , 'generated*_fromPairs.csv'))[0])
    i=0
    ps, cs, tot = [], [], []
    for i in data.index:
        if i ==0: i=i+1
        if(print_stuff):
            print(i)
        latent_protein = None
        latent_cell = None
        if 'protein' in data:
            print([data.loc[i, 'protein']])
            latent_protein = model.encode_protein(protein=[data.loc[i, 'protein']], batch_size=1)
            ps.append(latent_protein.numpy())
            if(print_stuff):
                print(latent_protein.shape)
        if 'cell_line' in data:
            latent_cell = model.encode_cell_line(cell_line=[data.loc[i, 'cell_line']], batch_size=1)
            cs.append(latent_cell.numpy())
            if(print_stuff):
                print(latent_cell.shape)
        if None not in (latent_protein, latent_cell):
            latent_z = model.together(latent_cell, latent_protein)
            tot.append(latent_z.numpy())
            print(latent_z.shape)
        if(print_stuff):
            if(i==5):
                print(ps, len(ps))
                1/0
    df = pd.DataFrame(
        {
            'SMILES': data['SMILES']
        }
    )
    if 'protein' in data:
        df['protein']= data['protein']
        df['latent_ptotein'] = ps
    if 'cell_line' in data:
        df['cell_line'] = data['cell_line']
        df['latent_cell'] = cs
    if None not in (latent_protein, latent_cell):
        df['latent_combined'] = tot
    df.to_csv(os.path.join(model.model_path, 'results', 'latent_fromPairs.csv'))
    # for protein, cell in zip(data['protein'], data['cell_line']):
    #     print(protein, cell)
    #     latent_cell = model.encode_cell_line(cell_line=[cell], batch_size=1)
    #     latent_protein = model.encode_protein(protein=[protein], batch_size=1)
    #     latent_z = model.together(latent_cell, latent_protein)
        
