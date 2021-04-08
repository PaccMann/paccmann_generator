import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import sys
import json
import logging
import torch
from paccmann_generator.utils import disable_rdkit_logging, add_avg_profile
from paccmann_generator.drug_evaluators.aromatic_ring import AromaticRing
from paccmann_generator.drug_evaluators.esol import ESOL
from paccmann_generator.drug_evaluators.molecular_weight import MolecularWeight
from paccmann_generator.drug_evaluators.qed import QED
from paccmann_generator.drug_evaluators.sas import SAS
from paccmann_generator.drug_evaluators.scsore import SCScore
from paccmann_generator.drug_evaluators.penalized_logp import PenalizedLogP
from paccmann_generator.drug_evaluators.tox21 import Tox21
from paccmann_generator.model import Model
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

with open(params_path) as f:
    params.update(json.load(f))

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
#model_name = model_name + '_'+ test_cell_line + '_lern' + str(params['learning_rate']) #+'_aromaticity' + str(params['aromaticity_weight'])
#logger.info(f'Model with name {model_name} starts.')

# Load protein sequence data
protein_df = pd.read_csv(protein_data_path, index_col=0)#, header=None, names=[str(x) for x in range(768)]) #'entry_name')
protein_df = protein_df[~protein_df.index.isnull()]
protein_df.index = [i.split('|')[2] for i in protein_df.index]
protein_seq_df = pd.read_csv(protein_data_seq_path, names = ['sequence'], index_col=0) #, index_col='entry_name')
#print(protein_seq_df.head)
protein_seq_df.index = [i.split('|')[2] for i in protein_seq_df.index]
protein_df = pd.concat([protein_df, protein_seq_df], axis=1, join='outer')
protein_df = protein_df[[s.split('_')[0] in cancer_genes for s in protein_df.index]]

name='liver_average_sanitize_SNU-423_lern0.0001_2_'
end = ['protein'] #'combined', 'omics', 
tox = Tox21(model_path = "/home/tol/Tox21_deepchem")

def get_C_fraction(smiles):
        """get the fraction of C atoms in the molecule

        Args:
            smiles (list): A list of SMILES strings.

        Returns:
            list: a list of the fractions of C atmons per molecule.
        """
        C=0
        if smiles:
            if len(smiles) is not 0:
                C = [1 for i in smiles if i=='C' or i=='c'].count(1)
                tot = Chem.MolFromSmiles(smiles).GetNumAtoms()
        return C/tot

def main(model_name, e):    
    #mols = pd.read_csv('biased_models/'+model_name+'/results/generated.csv').drop(columns=['Unnamed: 0']) #_per_epoch
    mols = pd.read_csv(model_name+'/valid_reactions.csv').drop(columns=['Unnamed: 0'])
    #loss = pd.read_csv('biased_models/'+model_name+'/results/loss_reward_evolution.csv').drop(columns=['Unnamed: 0']) #_and_more
    model = Model('omics', params, omics_df, protein_df, logger)
    mols['C_frac'] = [np.nan]*mols.shape[0]
    mols['aromatic'] = [np.nan]*mols.shape[0]
    mols['esol'] = [np.nan]*mols.shape[0]
    mols['mol_weight'] = [np.nan]*mols.shape[0]
    mols['qed'] = [np.nan]*mols.shape[0]
    mols['sas'] = [np.nan]*mols.shape[0]
    mols['SCScore'] = [np.nan]*mols.shape[0]
    mols['penalized_logP'] = [np.nan]*mols.shape[0]
    mols['tox_mean'] = [np.nan]*mols.shape[0]
    mols['tox_frac'] = [np.nan]*mols.shape[0]
    mols['NrAromRing'] = [np.nan]*mols.shape[0]
    for idx, s in enumerate(mols['SMILES']):
        if(s != s): mol=None
        else: mol = Chem.MolFromSmiles(s)
        if(mol): smile = Chem.MolToSmiles(mol)
        else: smile=None
        if(smile):
            mols.loc[idx, 'C_frac'] = get_C_fraction(s)
            mols.loc[idx, 'aromatic'] = arom(s)
            mols.loc[idx, 'esol'] = esol(s)
            mols.loc[idx, 'mol_weight'] = mol_weight(s)/100
            mols.loc[idx, 'qed'] = qed(s)
            mols.loc[idx, 'sas'] = sas(s)
            mols.loc[idx, 'SCScore'] = scscore(s)
            mols.loc[idx, 'penalized_logP'] = penalized_logp(s)
            for i in omics_df.cell_line:
                mols.loc[idx, 'IC50_'+i] = get_IC50(s, model.model , i)
            tox_res = get_tox(s)
            mols.loc[idx, 'tox_mean'] = tox_res[0]
            mols.loc[idx, 'tox_frac'] = tox_res[1]
            mols.loc[idx, 'NrAromRing'] = NrAromRing(s)
        else:
            mols.loc[idx, 'C_frac'] = np.nan
            mols.loc[idx, 'aromatic'] = np.nan
            mols.loc[idx, 'esol'] = np.nan
            mols.loc[idx, 'mol_weight'] = np.nan
            mols.loc[idx, 'qed'] = np.nan
            mols.loc[idx, 'sas'] = np.nan
            mols.loc[idx, 'SCScore'] = np.nan
            mols.loc[idx, 'penalized_logP'] = np.nan
            mols.loc[idx, 'IC50'] = np.nan
            mols.loc[idx, 'tox_mean'] = np.nan
            mols.loc[idx, 'tox_frac'] = np.nan
            mols.loc[idx, 'NrAromRing'] = np.nan
    
    mols.to_csv(model_name+'/valid_reactions_properties.csv')
    # mols.to_csv('biased_models/'+model_name+'/results/generated_properties.csv')
    # mols.describe().to_csv('biased_models/'+model_name+'/results/Properties_overview.csv')
    # mols = mols.groupby(['epoch']).mean()
    # #loss['C_frac'] = [np.nan]*loss.shape[0]
    # #loss['aromatic'] = [np.nan]*loss.shape[0]
    # loss = loss.join(mols, on='epoch') 
    # print(loss)
    # loss.to_csv('biased_models/'+model_name+'/results/loss_and_metrics.csv')

def get_IC50(smiles, model, cell_line):
    """prints the IC50 of of compounds and cell_line

    Args:
        smiles (string): the smiles representation of a compound
        model (reinforce object): the trained model.
        cell_line (string): the cell line
    """
    # smiles= [smiles, smiles]
    # #cell:
    # #print(cell_line)
    # #print(np.sum(self.gep_df['cell_line'].isin(cell_line)), "iloc0 \n", self.gep_df[self.gep_df['cell_line'].isin(cell_line)]['gene_expression'])
    # cell_mu = []
    # cell_logvar = []
    # gep_ts = []
    # batch_size = len(smiles)
    # for cell in [cell_line]:
    #     gep_t = torch.unsqueeze(
    #         torch.Tensor(
    #             model.gep_df[
    #                 model.gep_df['cell_line'] == cell  # yapf: disable
    #             ].iloc[0].gene_expression.values
    #         ),
    #         0
    #     )
    #     gep_ts.append(torch.unsqueeze(gep_t,0).detach().numpy()[0][0])
    #     cell_mu_i, cell_logvar_i = model.encoder_omics(gep_t)
    #     #print(torch.unsqueeze(cell_mu_i, 0).detach().numpy())
    #     cell_mu.append(torch.unsqueeze(cell_mu_i, 0).detach().numpy()[0][0])
    #     cell_logvar.append(torch.unsqueeze(cell_logvar_i, 0).detach().numpy()[0][0])
    # gep_ts = torch.as_tensor(gep_ts)
    # cell_mu = torch.as_tensor(cell_mu)
    # cell_logvar = torch.as_tensor(cell_logvar)
    # #print("before", cell_mu.size())
    # #print(cell_mu)
    # cell_mu_batch = cell_mu.repeat(batch_size, 1)
    # cell_logvar = cell_logvar.repeat(batch_size, 1)
    # gep_ts = gep_ts.repeat(batch_size, 1)
    # if cell_mu_batch.size()[0]>batch_size:
    #     cell_mu_batch = cell_mu_batch[:batch_size]
    #     cell_logvar = cell_logvar[:batch_size]
    #     gep_ts = gep_ts[:batch_size]

    # smiles_t_efficacy = model.smiles_to_numerical(smiles, target='efficacy')

    # # Evaluate drugs
    # predO, pred_dictO = model.efficacy_predictor(
    #     smiles_t_efficacy, gep_ts
    # )
    # log_preds = model.get_log_molar(np.squeeze(predO.detach().numpy()))
    log_preds = model.get_reward_paccmann([smiles, smiles], [cell_line], [True, True], 2, print_log=True)
    return log_preds[0]


def get_tox(smiles):
    """prints the toxicity (mean tox and fraction of tox > 0.5) of the compounds

    Args:
        smiles (string): the smiles representation of a compound
    """
    smiles_tensor = tox.preprocess_smiles(smiles)
    predictions, _ = tox.model(smiles_tensor)
    predictions = predictions[0, :].detach().numpy()
    pred = np.mean(predictions)
    high = (predictions >= 0.5).sum()
    return pred, high/len(predictions)

def NrAromRing(smiles): 
    return Chem.rdMolDescriptors.CalcNumAromaticRings(Chem.MolFromSmiles(smiles))

def main2(model):   
    folder= '/home/tol/data/'+model+'/results/'
    file_name = folder+'generated.csv'
    model = Model('average', params, omics_df, protein_df, logger)
    mols = pd.read_csv(file_name)
    # with open(file_name, "r") as ins:
    #     smiles = []
    #     proteins = []
    #     for line in ins:
    #         content = line.split('\t')
    #         smiles.append(content[0])
    #         proteins.append(content[1].split('\n')[0])
    # mols = pd.DataFrame(smiles, columns=['SMILES'])
    # mols['protein'] = proteins
    mols['C_frac'] = [np.nan]*mols.shape[0]
    mols['aromatic'] = [np.nan]*mols.shape[0]
    mols['esol'] = [np.nan]*mols.shape[0]
    mols['mol_weight'] = [np.nan]*mols.shape[0]
    mols['qed'] = [np.nan]*mols.shape[0]
    mols['sas'] = [np.nan]*mols.shape[0]
    mols['SCScore'] = [np.nan]*mols.shape[0]
    mols['penalized_logP'] = [np.nan]*mols.shape[0]
    mols['tox_mean'] = [np.nan]*mols.shape[0]
    mols['tox_frac'] = [np.nan]*mols.shape[0]
    mols['NrAromRing'] = [np.nan]*mols.shape[0]
    for idx, s in enumerate(mols['SMILES']):
        if(s != s): mol=None
        else: mol = Chem.MolFromSmiles(s)
        if(mol): smile = Chem.MolToSmiles(mol)
        else: smile=None
        if(smile):
            mols.loc[idx, 'C_frac'] = get_C_fraction(s)
            mols.loc[idx, 'aromatic'] = arom(s)
            mols.loc[idx, 'esol'] = esol(s)
            mols.loc[idx, 'mol_weight'] = mol_weight(s)
            mols.loc[idx, 'qed'] = qed(s)
            mols.loc[idx, 'sas'] = sas(s)
            mols.loc[idx, 'SCScore'] = scscore(s)
            mols.loc[idx, 'penalized_logP'] = penalized_logp(s)
            for i in omics_df.cell_line:
                mols.loc[idx, 'IC50_'+i] = get_IC50(s, model.model , i)
            tox_res = get_tox(s)
            mols.loc[idx, 'tox_mean'] = tox_res[0]
            mols.loc[idx, 'tox_frac'] = tox_res[1]
            mols.loc[idx, 'NrAromRing'] = NrAromRing(s)
            ## add IC50 and tox
        else:
            mols.loc[idx, 'C_frac'] = np.nan
            mols.loc[idx, 'aromatic'] = np.nan
            mols.loc[idx, 'esol'] = np.nan
            mols.loc[idx, 'mol_weight'] = np.nan
            mols.loc[idx, 'qed'] = np.nan
            mols.loc[idx, 'sas'] = np.nan
            mols.loc[idx, 'SCScore'] = np.nan
            mols.loc[idx, 'penalized_logP'] = np.nan
            mols.loc[idx, 'IC50'] = np.nan
            mols.loc[idx, 'tox_mean'] = np.nan
            mols.loc[idx, 'tox_frac'] = np.nan
            mols.loc[idx, 'NrAromRing'] = np.nan
    print(mols.describe(), mols)
    mols.to_csv(folder+'/generated_properties.csv')

arom = AromaticRing()
esol = ESOL()
mol_weight = MolecularWeight()
qed = QED()
sas = SAS()
scscore = SCScore()
penalized_logp = PenalizedLogP()
end = ['onlyConcat'] #'set', 'average', 
for e in end:
    model = 'liver_'+ e + '_sanitized_SNU-423_lern0.0001_combined2'
    model = '../data/results/HB_HuH-7_generated2'
    main(model, e)