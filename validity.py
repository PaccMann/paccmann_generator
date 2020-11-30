from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd

def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            print("1",smiles_or_mol)
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            print("2",smiles_or_mol)
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            print("3",smiles_or_mol)
            return None
        return mol
    return smiles_or_mol

def valid_smiles(smiles):
    if True:
        imgs = []
        for s in smiles:
            if(s):
                #print(s)
                imgs.append(Chem.MolFromSmiles(s, sanitize= True))
        # imgs = [Chem.MolFromSmiles(s, sanitize=True) for s in smiles]
    else:
        imgs = [Chem.MolFromSmiles(s, sanitize=False) for s in smiles]
    valid_idxs = [ind for ind in range(len(imgs)) if imgs[ind] is not None]
    # valid_idxs = [ind for ind, img in enumerate(imgs) if img is not None]

    smiles = [
        smiles[ind] for ind in range(len(smiles)) if ind not in valid_idxs
    ]
    print("own:", smiles)
    return smiles

def MOSES(smiles):
    print("other:")
    return [get_mol(s) for s in smiles]


smiles=['C']
MOSES(smiles)
valid_smiles(smiles)
#print(get_mol(smiles), valid_smiles(smiles))
data = pd.read_csv('biased_models/liver_average_sanitized_SNU-423_omics/generated_smiles_2_fromPairs.csv')
smiles = data['SMILES'].tolist()
i=3
a = MOSES(smiles[1:]) 
b = valid_smiles(smiles[1:])
mol= Chem.MolFromSmiles('C2ccc(C)cc2(SC(=O)CC1CCCC1C)=O')
for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
print(Chem.MolToSmiles(mol))
Draw.MolToFile(mol,'biased_models/liver_average_sanitized_SNU-423_omics/test_mol.png')
