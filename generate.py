import argparse
import json
import logging
import os
import sys
import numpy as np
import pandas as pd
import warnings
import torch
from numpy import savetxt
import glob
warnings.filterwarnings("ignore")
from paccmann_chemistry.models import (
    StackGRUDecoder, StackGRUEncoder, TeacherVAE
)
from rdkit.Chem import Descriptors
from rdkit import Chem
from paccmann_generator.drug_evaluators.aromatic_ring import AromaticRing
from paccmann_generator.drug_evaluators.esol import ESOL
from paccmann_generator.drug_evaluators.qed import QED
from paccmann_generator.drug_evaluators.sas import SAS
from paccmann_generator.drug_evaluators.scsore import SCScore
from paccmann_generator.drug_evaluators.penalized_logp import PenalizedLogP
from paccmann_chemistry.utils import get_device
from paccmann_generator.plot_utils import plot_and_compare, plot_and_compare_proteins, plot_loss
from paccmann_generator.utils import add_avg_profile, omics_data_splitter, protein_data_splitter
from paccmann_omics.encoders import ENCODER_FACTORY
from pytoda.smiles.smiles_language import SMILESLanguage
from pytoda.proteins.protein_language import ProteinLanguage
from paccmann_generator.utils import disable_rdkit_logging
from paccmann_predictor.models import MODEL_FACTORY as MODEL_FACTORY_OMICS
import sys
#sys.path.append('/dataP/tol/paccmann_affinity')
sys.path.append('/home/tol/paccmann_affinity')
from paccmann_affinity.models.predictors import MODEL_FACTORY as MODEL_FACTORY_PROTEIN

from paccmann_generator.reinforce_proteins_omics import ReinforceProteinOmics
from paccmann_generator import ReinforceOmic
from paccmann_generator.reinforce_proteins import ReinforceProtein
from files import *
cancer_cell_lines = ['HUH-6-clone5','HuH-7','SNU-475','SNU-423','SNU-387','SNU-449','HLE','C3A']

def generate():
    batch_size1 = 150
    batch_size = 1000
    proteins = protein_df.index.tolist()
    cell_line = [test_cell_line]
    valid_smiles_batch_combined = []
    valid_smiles_batch_omics = []
    valid_smiles_batch_proteins = []
    first_iter = None
    p, c = [], []
    # while(len(valid_smiles_batch_combined)<batch_size):
    #     combined.generate_len = batch_size1
    #     valid_smiles_c, idx = combined.generate_compound(
    #                 batch_size1, proteins, cell_line
    #     )
    #     valid_smiles_batch_combined = np.append(valid_smiles_batch_combined, valid_smiles_c)
    #     p = np.append(p, [val for i, val in enumerate(proteins*batch_size1) if i in idx])
    #     c = np.append(c, [val for i, val in enumerate(cell_line*batch_size1) if i in idx])
    #     #print(len(valid_smiles_batch_combined), len(p), len(c))
    #     print(len(valid_smiles_batch_combined))
    # df = pd.DataFrame(
    #     {
    #         'protein': p,
    #         'cell_line': c,
    #         'SMILES': valid_smiles_batch_combined
    #     }
    # )
    # df.to_csv(combined.model_path+"/generated_smiles_"+gen_epoch+"_fromPairs_withoutEmpty.csv")
    #savetxt(combined.model_path+"/generated_smiles_"+gen_epoch+"_fromPairs.csv", valid_smiles_batch_combined, delimiter=',', fmt=('%s'))
    c = []
    while(len(valid_smiles_batch_omics)<batch_size):
        omics.generate_len = batch_size1
        valid_smiles_o, idx = omics.generate_compound(
                    batch_size1, cell_line
        )
        valid_smiles_batch_omics = np.append(valid_smiles_batch_omics, valid_smiles_o)
        c = np.append(c, [val for i, val in enumerate(cell_line*batch_size1) if i in idx])
        print(len(valid_smiles_batch_omics))
    df = pd.DataFrame(
        {
            'cell_line': c,
            'SMILES': valid_smiles_batch_omics
        }
    )
    df.to_csv(omics.model_path+"/generated_smiles_"+omics_epoch+"_fromPairs_withoutEmpty.csv")
    #savetxt(omics.model_path+"/generated_smiles_"+omics_epoch+"_fromPairs.csv", valid_smiles_batch_omics, delimiter=',', fmt=('%s'))
    # p = []
    # while(len(valid_smiles_batch_proteins)<batch_size):
    #     protein.generate_len = batch_size1
    #     valid_smiles_p, idx = protein.generate_compound(
    #                 batch_size1, proteins
    #     )
    #     #print(len(valid_smiles_batch), len(valid_smiles))
    #     valid_smiles_batch_proteins = np.append(valid_smiles_batch_proteins, valid_smiles_p)
    #     p = np.append(p, [val for i, val in enumerate(proteins*batch_size1) if i in idx])
    #     print(len(valid_smiles_batch_proteins))
    #     # print(len(idx_batch))
    # df = pd.DataFrame(
    #     {
    #         'protein': p,
    #         'SMILES': valid_smiles_batch_proteins
    #     }
    # )
    # df.to_csv(protein.model_path+"/generated_smiles_"+protein_epoch+"_fromPairs_withoutEmpty.csv")
    #savetxt(protein.model_path+"/generated_smiles_"+protein_epoch+"_fromPairs.csv", valid_smiles_batch_proteins, delimiter=',', fmt=('%s'))

def get_C_fraction(smiles):
        """get the fraction of C atoms in the molecule

        Args:
            smiles (list): A list of SMILES strings.

        Returns:
            list: a list of the fractions of C atmons per molecule.
        """
        C=0
        tot = 1
        if not smiles is np.nan:
            if len(smiles) is not 0:
                C = [1 for i in smiles if i=='C' or i=='c'].count(1)
                tot = Chem.MolFromSmiles(smiles).GetNumAtoms()
        return C/tot

arom = AromaticRing()
esol = ESOL()
qed = QED()
sas = SAS()
scscore = SCScore()
penalized_logp = PenalizedLogP()

def get_metrics():
    data = pd.read_csv(omics.model_path+"/generated_smiles_"+omics_epoch+"_fromPairs_withoutEmpty.csv", index_col = 0)
    # data = data.iloc[:10, :]
    # print(data.shape)
    C_frac = []
    aroms, esols, qeds, sass, sc, logp, molWt, lens = [],[], [], [], [], [], [], []
    for i in data['SMILES']:
        if i is not np.nan:
            C_frac.append(get_C_fraction(i))
            aroms.append(arom(Chem.MolFromSmiles(i)))
            esols.append(esol(Chem.MolFromSmiles(i)))
            qeds.append(qed(Chem.MolFromSmiles(i)))
            sass.append(sas(Chem.MolFromSmiles(i)))
            sc.append(scscore(Chem.MolFromSmiles(i)))
            logp.append(penalized_logp(Chem.MolFromSmiles(i)))
            molWt.append(Descriptors.MolWt(Chem.MolFromSmiles(i)))
            lens.append(Chem.MolFromSmiles(i).GetNumAtoms())
        else:
            C_frac.append(np.nan)
            aroms.append(np.nan)
            esols.append(np.nan)
            qeds.append(np.nan)
            sass.append(np.nan)
            sc.append(np.nan)
            logp.append(np.nan)
            molWt.append(np.nan)
            lens.append(np.nan)

    data['C_fraction'] = C_frac
    data['aromaticity'] = aroms
    data['esol'] = esols
    data['qed'] = qeds
    data['sas'] = sass
    data['scscore'] = sc
    data['penalized_logp'] = logp
    data['MolWt'] = molWt
    data['len'] = lens
    print(data.head())
    data.to_csv(omics.model_path+'/results/generated_test_smiles_and_metrics.csv')

def latent_space(epoch_nr):
    models = [protein, omics] # combined, 
    print_stuff=False
    for model in models:
        data = pd.read_csv(glob.glob(os.path.join(model.model_path , 'generated*_fromPairs_withoutEmpty.csv'))[0])
        i=0
        ps, cs, tot = [], [], []
        for i in data.index:
            #if i ==0: i=i+1
            if(print_stuff):
                print(i)
            latent_protein = None
            latent_cell = None
            if 'protein' in data:
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
                tot.append(latent_z.detach().numpy())
                if(print_stuff):
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
        df.to_csv(os.path.join(model.model_path, 'results', 'latent_fromPairs_withoutEmpty.csv'))

    for idx, model in enumerate(models):
        data = pd.read_csv(os.path.join(model.model_path , 'results/generated_per_epoch.csv'))
        data = data[data.epoch == int(epoch_nr[idx])]
        data = data[data['mode'] == 'train']
        i=0
        ps, cs, tot = [], [], []
        for i in data.index:
            #if i ==0: i=i+1
            if(print_stuff):
                print(i)
            latent_protein = None
            latent_cell = None
            if 'protein' in data:
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
                tot.append(latent_z.detach().numpy())
                if(print_stuff):
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
        df.to_csv(os.path.join(model.model_path, 'results', 'latent_training.csv'))

# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('train_paccmann_rl')
logger_m = logging.getLogger('matplotlib')
logger_m.setLevel(logging.WARNING)

disable_rdkit_logging()

params = dict()
params['site'] = site
params['cancertype'] = cancertype

with open(params_path) as f:
    params.update(json.load(f))

logger.info(f'Model with name {model_name} starts.')

omics_df = pd.read_pickle(omics_data_path)
omics_df = add_avg_profile(omics_df)
idx = [i in cancer_cell_lines for i in omics_df['cell_line']]
omics_df  = omics_df[idx]
print("omics data:", omics_df.shape, omics_df['cell_line'].iloc[0])
test_cell_line = omics_df['cell_line'].iloc[6]

model_name = 'average_sanitized'
remove_invalid = True
gen_epoch = "13"
omics_epoch = "3"
protein_epoch = "29" 
model_name = model_name + '_' + test_cell_line +'_lern' + str(params['learning_rate'])

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

model_folder_name = site + '_' + model_name + '_combined'
# combined = ReinforceProteinOmics(generator, protein_encoder, cell_encoder, \
#     protein_predictor, paccmann_predictor, protein_df, omics_df, \
#     params, generator_smiles_language, model_folder_name, logger, remove_invalid
# )
# combined.load("gen_"+gen_epoch+".pt", "enc_"+gen_epoch+"_protein.pt"
# , "enc_"+gen_epoch+"_omics.pt")
# #combined.eval()


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
omics = ReinforceOmic(
    generator_rl_o, cell_encoder_rl_o, paccmann_predictor, omics_df, params,
    generator_smiles_language, model_folder_name, logger, remove_invalid
)
omics.load("gen_"+omics_epoch+".pt", "enc_"+omics_epoch+".pt")
#omics.eval()

# gru_encoder_rl_p = StackGRUEncoder(mol_params)
# gru_decoder_rl_p = StackGRUDecoder(mol_params)
# generator_rl_p = TeacherVAE(gru_encoder_rl_p, gru_decoder_rl_p)
# generator_rl_p.load(
#     os.path.join(
#         mol_model_path, f"weights/best_{params.get('metric', 'rec')}.pt"
#     ),
#     map_location=get_device()
# )
# generator_rl_p.eval()
# #generator_rl._associate_language(generator_smiles_language)

# protein_encoder_rl_p = ENCODER_FACTORY['dense'](protein_params)
# protein_encoder_rl_p.load(
#     os.path.join(
#         protein_model_path,
#         f"weights/best_{params.get('metric', 'both')}_encoder.pt"
#     ),
#     map_location=get_device()
# )
# protein_encoder_rl_p.eval()

# model_folder_name = site + '_' + model_name + '_protein'
# protein = ReinforceProtein(
#     generator_rl_p, protein_encoder_rl_p, protein_predictor, protein_df, params,
#     generator_smiles_language, model_folder_name, logger, remove_invalid
# )
# protein.load("gen_"+protein_epoch+".pt", "enc_"+protein_epoch+".pt")
#protein.eval()
generate()
get_metrics()
#latent_space([protein_epoch, omics_epoch])

