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
from paccmann_generator.drug_evaluators.molecular_weight import MolecularWeight
from paccmann_generator.drug_evaluators.qed import QED
from paccmann_generator.drug_evaluators.sas import SAS
from paccmann_generator.drug_evaluators.scsore import SCScore
from paccmann_generator.drug_evaluators.penalized_logp import PenalizedLogP
from paccmann_generator.utils import disable_rdkit_logging
from paccmann_generator.model import Model
import sys
sys.path.append('/home/tol/paccmann_affinity')
from files import *
cancer_cell_lines = ['HUH-6-clone5','HuH-7','SNU-475','SNU-423','SNU-387','SNU-449','HLE','C3A']

def generate(protein_df, test_cell_line, model, batch_size=50000):
    """generate more compounds from a certain epoch.

    Args:
        protein_df (DataFrame): Protein Data
        test_cell_line (array): cell lines to use
        model (a reinforce model): a combined or single model
        batch_size (int, optional): the number of compounds to generate. Defaults to 50000.
    """
    batch_steps = 150
    proteins = protein_df.index.tolist()
    cell_line = [test_cell_line]
    log_preds = []
    preds = []
    valid_smiles_batch = []
    p, c = [], []
    while(len(valid_smiles_batch)<batch_size):
        if(proteins is None):
            valid_smiles_c, predP, log_predsO, idx = model.generate_compounds_and_evaluate(
                        None, batch_steps, cell_line=cell_line
            )
        elif(cell_line is None):
            valid_smiles_c, predP, log_predsO, idx = model.generate_compounds_and_evaluate(
                        None, batch_steps, protein=proteins
            )
        else:
            valid_smiles_c, predP, log_predsO, idx = model.generate_compounds_and_evaluate(
                        None, batch_steps, protein=proteins, cell_line=cell_line
            )
        valid_smiles_batch = np.append(valid_smiles_batch, valid_smiles_c)
        p = np.append(p, [val for i, val in enumerate(proteins*batch_steps) if i in idx])
        c = np.append(c, [val for i, val in enumerate(cell_line*batch_steps) if i in idx])
        log_preds = np.append(log_preds, log_predsO)
        preds = np.append(preds, predP)
        print(len(valid_smiles_batch))
    df = pd.DataFrame(
        {
            'protein': p,
            'cell_line': c,
            'SMILES': valid_smiles_batch,
            'IC50': log_preds,
            'affinity';preds
        }
    )
    df.to_csv(model.model_path+"/generated_smiles_"+comb_epoch+".csv")
    
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

tox = Tox21(model_path = "/home/tol/Tox21_deepchem")
arom = AromaticRing()
esol = ESOL()
qed = QED()
sas = SAS()
scscore = SCScore()
penalized_logp = PenalizedLogP()

def get_IC50(smiles, model, cell_line):
    """prints the IC50 of of compounds and cell_line

    Args:
        smiles (string): the smiles representation of a compound
        model (reinforce object): the trained model.
        cell_line (string): the cell line
    """
    log_preds = model.get_reward_paccmann([smiles, smiles], [cell_line], [True, True], 2, print_log=True)
    return log_preds[0]

def get_metrics():
    data = pd.read_csv(omics.model_path+"/generated_smiles_"+comb_epoch+".csv", index_col = 0)
    C_frac = []
    aroms, esols, qeds, sass, sc, logp, molWt, lens = [],[], [], [], [], [], [], []
    for idx, i in enumerate(data['SMILES']):
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
            for c in omics_df.cell_line:
                data.loc[idx, 'IC50_'+c] = get_IC50(i, model.model , c)
            tox_res = get_tox(i)
            data.loc[idx, 'tox_mean'] = tox_res[0]
            data.loc[idx, 'tox_frac'] = tox_res[1]
            data.loc[idx, 'NrAromRing'] = NrAromRing(i)
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
    data.to_csv(omics.model_path+'/results/generated_smiles_"+comb_epoch+"_metrics.csv')

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
test_cell_line ='HuH-7'#omics_df['cell_line'].iloc[0]

model_name = 'average_sanitized'
remove_invalid = True
comb_epoch = "37" 
model_name = model_name + '_' + test_cell_line +'_lern' + str(params['learning_rate'])

protein_df = pd.read_csv(protein_data_path, index_col=0)
protein_df = protein_df[~protein_df.index.isnull()]
protein_df.index = [i.split('|')[2] for i in protein_df.index]
protein_seq_df = pd.read_csv(protein_data_seq_path, names = ['sequence'], index_col=0)
protein_seq_df.index = [i.split('|')[2] for i in protein_seq_df.index]
protein_df = pd.concat([protein_df, protein_seq_df], axis=1, join='outer')
protein_df = protein_df[[s.split('_')[0] in cancer_genes for s in protein_df.index]]
print("proteins:", protein_df.index, len(cancer_genes))

# load the model of the specified epoch
model_folder_name = site + '_' + model_name + '_combined'
model = Model('average', params, omics_df, protein_df, logger, model_folder_name)
model.model.load("gen_"+comb_epoch+".pt", "enc_"+comb_epoch+"_protein.pt", "enc_"+comb_epoch+"_omics.pt")
model.model.eval()

generate(protein_df, test_cell_line, model.model)
get_metrics()

