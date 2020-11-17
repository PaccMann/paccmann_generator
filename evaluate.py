import pandas as pd
import numpy as np
model='liver_concat_allValid_SNU-423_combined'
mols = pd.read_csv('biased_models/'+model+'/results/generated.csv', index_col=0)
loss = pd.read_csv('biased_models/'+model+'/results/loss_reward_evolution.csv', index_col=0)

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

loss['C_frac'] = [np.nan]*loss.shape[0]
loss['C_frac'] = [np.nan]*loss.shape[0]
for idx, s in enumerate(mols['SMILES']):
    loss.loc[idx, 'C_frac'] = get_C_fraction(s)

print(mols.head(), loss.head(), loss.columns, loss.shape, mols.shape)