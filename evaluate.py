import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from paccmann_generator.drug_evaluators.aromatic_ring import AromaticRing
from paccmann_generator.drug_evaluators.esol import ESOL
from paccmann_generator.drug_evaluators.molecular_weight import MolecularWeight
from paccmann_generator.drug_evaluators.qed import QED
from paccmann_generator.drug_evaluators.sas import SAS
from paccmann_generator.drug_evaluators.scsore import SCScore
from paccmann_generator.drug_evaluators.penalized_logp import PenalizedLogP
name='liver_average_allValid_temp08_SNU-423_'
end = ['combined', 'omics', 'protein'] #

def get_C_fraction(smiles, model):
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

def main(model):    
    mols = pd.read_csv('biased_models/'+model+'/results/generated.csv').drop(columns=['Unnamed: 0']) #_per_epoch
    loss = pd.read_csv('biased_models/'+model+'/results/loss_reward_evolution.csv').drop(columns=['Unnamed: 0']) #_and_more
    mols['C_frac'] = [np.nan]*mols.shape[0]
    mols['aromatic'] = [np.nan]*mols.shape[0]
    mols['esol'] = [np.nan]*mols.shape[0]
    mols['mol_weight'] = [np.nan]*mols.shape[0]
    mols['qed'] = [np.nan]*mols.shape[0]
    mols['sas'] = [np.nan]*mols.shape[0]
    mols['SCScore'] = [np.nan]*mols.shape[0]
    mols['penalized_logP'] = [np.nan]*mols.shape[0]
    for idx, s in enumerate(mols['SMILES']):
        if(s != s): mol=None
        else: mol = Chem.MolFromSmiles(s)
        if(mol): smile = Chem.MolToSmiles(mol)
        else: smile=None
        if(smile):
            mols.loc[idx, 'C_frac'] = get_C_fraction(s, model)
            mols.loc[idx, 'aromatic'] = arom(s)
            mols.loc[idx, 'esol'] = esol(s)
            mols.loc[idx, 'mol_weight'] = mol_weight(s)/100
            mols.loc[idx, 'qed'] = qed(s)
            mols.loc[idx, 'sas'] = sas(s)
            mols.loc[idx, 'SCScore'] = scscore(s)
            mols.loc[idx, 'penalized_logP'] = penalized_logp(s)
        else:
            mols.loc[idx, 'C_frac'] = np.nan
            mols.loc[idx, 'aromatic'] = np.nan
            mols.loc[idx, 'esol'] = np.nan
            mols.loc[idx, 'mol_weight'] = np.nan
            mols.loc[idx, 'qed'] = np.nan
            mols.loc[idx, 'sas'] = np.nan
            mols.loc[idx, 'SCScore'] = np.nan
            mols.loc[idx, 'penalized_logP'] = np.nan
    mols.describe().to_csv('biased_models/'+model+'/results/Properties_overview.csv')
    mols = mols.groupby(['epoch']).mean()
    #loss['C_frac'] = [np.nan]*loss.shape[0]
    #loss['aromatic'] = [np.nan]*loss.shape[0]
    loss = loss.join(mols, on='epoch') 
    print(loss)
    loss.to_csv('biased_models/'+model+'/results/loss_and_metrics.csv')

arom = AromaticRing()
esol = ESOL()
mol_weight = MolecularWeight()
qed = QED()
sas = SAS()
scscore = SCScore()
penalized_logp = PenalizedLogP()
for e in end:
    model = name + e
    main(model)