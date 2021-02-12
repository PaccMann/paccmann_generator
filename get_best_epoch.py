import pandas as pd
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interact, interactive, fixed, interact_manual
import numpy as np
from rdkit.Chem import Descriptors
from rdkit import Chem
from paccmann_generator.drug_evaluators.aromatic_ring import AromaticRing
from paccmann_generator.drug_evaluators.esol import ESOL
from paccmann_generator.drug_evaluators.qed import QED
from paccmann_generator.drug_evaluators.sas import SAS
from paccmann_generator.drug_evaluators.scsore import SCScore
from paccmann_generator.drug_evaluators.penalized_logp import PenalizedLogP
from paccmann_generator.drug_evaluators.tox21 import Tox21
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

model_name = 'average_sanitized'
remove_invalid = True
gen_epoch = "13"
omics_epoch = "4"
protein_epoch = "29" 

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
tox = Tox21(model_path = "/home/tol/Tox21_deepchem")

def get_metrics(file_path, file_name):
    """compute the metrics and save them in a file.

    Args:
        file_path (string): the path to the files
        file_name (string): the name of the file with the smiles.
    """
    data = pd.read_csv(file_path + file_name) #, index_col = 0)
    # data = data.iloc[:10, :]
    # print(data.shape)
    C_frac = []
    aroms, esols, qeds, sass, sc, logp, molWt, lens, ic50 = [],[], [], [], [], [], [], [], []
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
    data.to_csv(file_path+'generated_and_metrics.csv')


def get_IC50(file_path, file_name):
    """prints the IC50 of of compounds from the test_cell_line

    Args:
        file_path (string): the path to the files
        file_name (strin): the name of the file containing the smiles
    """
    data = pd.read_csv(file_path + file_name)
    for idx in data.index:
        if(data.loc[idx, 'cell_line']==test_cell_line):
            mol = data.loc[idx,'SMILES']
            print(type([mol]), [mol], test_cell_line)
            log_preds = omics.get_reward_paccmann([mol, mol], [test_cell_line], [True, True], 2)
            print(mol, log_preds)
    return 

def get_tox(file_path, file_name):
    """prints the toxicity of the compounds

    Args:
        file_path (string): the path to the files
        file_name (strin): the name of the file containing the smiles
    """
    data = pd.read_csv(file_path + file_name)
    for mol in data['SMILES']:
        toxicity = tox(mol)
        smiles_tensor = tox.preprocess_smiles(mol)
        predictions, _ = tox.model(smiles_tensor)
        predictions = predictions[0, :].detach().numpy()
        #print(predictions)
        pred = np.mean(predictions)
        high = (predictions >= 0.5).sum()
        print(mol, pred, high/len(predictions))
    return 

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

with open(params_path) as f:
    params.update(json.load(f))
omics_df = pd.read_pickle(omics_data_path)
omics_df = add_avg_profile(omics_df)
idx = [i in cancer_cell_lines for i in omics_df['cell_line']]
omics_df  = omics_df[idx]
print("omics data:", omics_df.shape)
test_cell_line = omics_df['cell_line'].iloc[0]
model_name = model_name + '_' + test_cell_line  + '_lern0.0001' #+ str(params['learning_rate'])

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
#combined.eval()


# gru_encoder_rl_o = StackGRUEncoder(mol_params)
# gru_decoder_rl_o = StackGRUDecoder(mol_params)
# generator_rl_o = TeacherVAE(gru_encoder_rl_o, gru_decoder_rl_o)
# generator_rl_o.load(
#     os.path.join(
#         mol_model_path, f"weights/best_{params.get('metric', 'rec')}.pt"
#     ),
#     map_location=get_device()
# )
# generator_rl_o.eval()
# generator_rl._associate_language(generator_smiles_language)

# cell_encoder_rl_o = ENCODER_FACTORY['dense'](cell_params)
# cell_encoder_rl_o.load(
#     os.path.join(
#         omics_model_path,
#         f"weights/best_{params.get('metric', 'both')}_encoder.pt"
#     ),
#     map_location=get_device()
# )
# cell_encoder_rl_o.eval()

# model_folder_name = site + '_' + model_name + '_omics'
# print("model:", model_folder_name)
# omics = ReinforceOmic(
#     generator_rl_o, cell_encoder_rl_o, paccmann_predictor, omics_df, params,
#     generator_smiles_language, model_folder_name, logger, remove_invalid
# )
# omics.load("gen_"+omics_epoch+".pt", "enc_"+omics_epoch+".pt")
# #omics.eval()

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

# model_folder_name = site + '_' + model_name + '_lern0.0001_C_frac0.8_protein'
# protein = ReinforceProtein(
#     generator_rl_p, protein_encoder_rl_p, protein_predictor, protein_df, params,
#     generator_smiles_language, model_folder_name, logger, remove_invalid
# )
# protein.load("gen_"+protein_epoch+".pt", "enc_"+protein_epoch+".pt")

for model in ['average']: # 'concat', 
    for part in ['lern0.0001_aromaticity'+ str(params['aromaticity_weight'])+'_combined']: #]: #,'omics', , 'combined'
        #get the right folders and files for the models
        print(model, part)
        file_name = 'smiles_hepatoblastoma_omics_metrics.csv'#'generated.csv'
        metrics_file = 'generated_and_metrics.csv'
        file_path = '/home/tol/data/'#'biased_models/liver_' + model + '_sanitized_'+test_cell_line+'_' + part + '/results/'
        print("read file", 'biased_models/liver_' + model + '_sanitized_'+test_cell_line+'_' + part + '/results/')
        #get the metrics
        #get_metrics(file_path, file_name)
        #get_metrics('~/data/', 'smiles_hepatoblastoma_omics.csv')
        #get_IC50('~/data/', 'smiles_hepatoblastoma_omics.csv')
        get_tox(file_path, file_name)
        1/0
        # file_path_coo = 'biased_models/liver_' + model + '_sanitized_SNU-423_' + part + '/results/grid_coordinates.csv'
        data = pd.read_csv(file_path + metrics_file, index_col = 0)
        reward = []
        rews, rew2 = [], []
        data['reward'] = [np.nan]*data.shape[0]
        for i in range(np.max(data['epoch'])):
            i=i+1
            smiles = data[data['epoch']==i]['SMILES']
            proteins = data[data['epoch']==i]['protein']
            valid_idx = [True]*len(data[data['epoch']==i]['SMILES'])
            batch_size = len(data[data['epoch']==i]['SMILES'])
            print(i)
            protein_predictor_tensor = []
            for prot in proteins:
                protein_encoder_tensor, protein_predictor_tensor_i = (
                    protein.protein_to_numerical(
                        prot, encoder_uses_sequence=False, predictor_uses_sequence=True
                    )
                )
                protein_predictor_tensor.append(torch.unsqueeze(protein_predictor_tensor_i, 0).detach().numpy()[0][0])
            protein_predictor_tensor = torch.as_tensor(protein_predictor_tensor)
            smiles_t = protein.smiles_to_numerical(smiles, target='predictor')
            pred, pred_dict = protein.affinity_predictor(
                smiles_t, protein_predictor_tensor[valid_idx]
            )
            rew = np.squeeze(pred.detach().numpy())
            rew2 = protein.get_reward_affinity(smiles, proteins, valid_idx, batch_size)
                #get_reward(data[data['epoch']==i]['SMILES'], data[data['epoch']==i]['protein'], 
                #[True]*len(data[data['epoch']==i]['SMILES']), len(data[data['epoch']==i]['SMILES']))
            print(len(rew))
            rews = np.concatenate((rews,rew))
            reward.append(np.mean(rew))
            # test = pd.DataFrame(data[data['epoch']==i]['Binding probability'])
            # test['rewards'] = rew2
            # test['pred'] = rews
            # print(test)
            # 1/0
        data['preds'] = rews
        
        columns = ['C_fraction', 'aromaticity', 'esol', 'qed', 'sas','scscore', 'penalized_logp', 'MolWt']
        #data[columns].plot()
        print(data.head())
        