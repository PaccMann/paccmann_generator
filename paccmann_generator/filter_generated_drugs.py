"""Filter generated drugs."""
import hashlib
import json
import os
import pandas as pd
from rdkit import Chem
from .drug_evaluators.esol import ESOL
from .drug_evaluators.molecular_weight import MolecularWeight
from .drug_evaluators.qed import QED
from .drug_evaluators.sas import SAS
from .drug_evaluators.tox21 import Tox21
from .drug_evaluators.organdb import OrganDB


def filter_generated_drugs(
    data_path,
    filters={
        'SAS': [1, 4],
        'QED': [0.3, 1],
        'ESOL': [-10, -2],
        'MolecularWeight': [0, 1000],
        'Tox21': [0, 0.4999],
        'OrganDB': [0, 0.4999],
    },
    overwrite_csv=False,
    csv_save=True,
    smi_save=False,
    returning=False,
):
    """
    Function to filter a list of generated drugs according to the specified
    criteria.

    Args:
        data_path (str): path to the drug data.
        filters (dict): filters to apply.
        overwrite_csv (bool): overwrite the .csv. Defaults to False.
        csv_save (bool): save filtered drugs in .csv format. Defaults to True.
        smi_save (bool): save filtered drugs in .smi format. Defaults to False.
        returing (bool): return the filtered drugs. Defaults to False.

    Returns:
        (pd.DataFrame or None): if returing is true returns the filtered drugs.
            None otherwise.
    """

    # Error handling
    if type(data_path) == pd.DataFrame:
        data = data_path
        del data_path
    elif type(data_path) == str:
        if data_path.endswith('.csv'):
            data = pd.read_csv(data_path)
        else:
            raise TypeError('The provided string does not point to .csv file')
    else:
        raise TypeError('1st argument should be a pd.DataFrame or a str')

    if 'SMILES' not in data:
        raise ValueError('No SMILES found in the dataframe.')
    if 'cell_line' not in data:
        data['cell_line'] = data_path.split('_avg')[0].split('/')[-1]

    if not type(filters) == dict:
        raise TypeError('Filters is not a dict.')
    if not all([type(v) == list and len(v) == 2 for v in filters.values()]):
        raise ValueError('Provide upper and lower bound for every filter')
    if (overwrite_csv or csv_save or smi_save) and 'data_path' not in locals():
        raise NameError('To save a file, provide a .csv as input, not a df.')

    # Allocate metrics
    sas = SAS()
    qed = QED()
    molecular_weight = MolecularWeight()
    esol = ESOL()
    tox21 = Tox21()
    organdb = OrganDB()

    # Compute scores
    molecules = data['SMILES'].apply(Chem.MolFromSmiles)
    data['QED'] = molecules.apply(qed)
    data['SAS'] = molecules.apply(sas)
    data['ESOL'] = molecules.apply(esol)
    data['MolecularWeight'] = molecules.apply(molecular_weight)
    data['Tox21'] = molecules.apply(tox21)
    data['OrganDB'] = molecules.apply(organdb)
    data['ID'] = data.apply(
        lambda row: hashlib.md5(
            f'{row["cell_line"]}-{row["SMILES"]}'.encode('utf-8')
        ).hexdigest(),
        axis=1,
    )

    if not all([k in data.columns for k in filters.keys()]):
        raise ValueError(
            'Filters contains an unknown key. Check documentation'
            ' to see an exhaustive list of allowed keys'
        )

    #  Filter molecules
    best = data
    for key in filters.keys():
        best = best[(best[key] > filters[key][0]) & (best[key] < filters[key][1])]

    # Saving options
    if overwrite_csv:
        data.to_csv(data_path)
    if csv_save or smi_save:  # Save json with filter conditions
        save_path = os.path.join('/', *data_path.split('/')[:-1])
        with open(os.path.join(save_path, 'filters.json'), 'w') as fp:
            json.dump(filters, fp)
    if csv_save:
        best.to_csv(os.path.join(save_path, 'best_mols.csv'))
    if smi_save:
        with open(os.path.join(save_path, 'best_mols.smi'), 'w') as fp:
            for row in best.iterrows():
                fp.write(row[1]['SMILES'] + '\t' + row[1].ID + '\n')

    if returning:
        return best
