import pandas as pd
import numpy as np
from rdkit import Chem
from paccmann_generator.drug_evaluators.aromatic_ring import AromaticRing
from paccmann_generator.drug_evaluators.esol import ESOL
from paccmann_generator.drug_evaluators.molecular_weight import MolecularWeight
model='liver_concat_allValid_SNU-423_combined'
mols = pd.read_csv('biased_models/'+model+'/results/generated_per_epoch.csv')
loss = pd.read_csv('biased_models/'+model+'/results/loss_and_more.csv')

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
                C = [1 for i in s if i=='C'].count(1)/len(smiles)
        return C
arom = AromaticRing()
esol = ESOL()
mol_weight = MolecularWeight()

mols = mols.drop(columns=['Unnamed: 0'])
mols['C_frac'] = [np.nan]*mols.shape[0]
mols['aromatic'] = [np.nan]*mols.shape[0]
mols['esol'] = [np.nan]*mols.shape[0]
mols['mol_weight'] = [np.nan]*mols.shape[0]
for idx, s in enumerate(mols['SMILES']):
    mol = Chem.MolFromSmiles(s)
    if(mol): smile = Chem.MolToSmiles(mol)
    else: smile=None
    if(smile):
        mols.loc[idx, 'C_frac'] = get_C_fraction(s)
        mols.loc[idx, 'aromatic'] = arom(s)
        mols.loc[idx, 'esol'] = esol(s)
        mols.loc[idx, 'mol_weight'] = mol_weight(s)
    else:
        mols.loc[idx, 'C_frac'] = np.nan
        mols.loc[idx, 'aromatic'] = np.nan
        mols.loc[idx, 'esol'] = np.nan
        mols.loc[idx, 'mol_weight'] = np.nan
print("describe",mols.describe())
mols = mols.groupby(['epoch']).mean()
print(mols)
#loss['C_frac'] = [np.nan]*loss.shape[0]
#loss['aromatic'] = [np.nan]*loss.shape[0]
loss = loss.join(mols, on='epoch') 
print(loss)
loss.to_csv('biased_models/'+model+'/results/loss_and_metrics.csv')