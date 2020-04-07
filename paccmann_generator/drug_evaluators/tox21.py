"""Tox21 evaluator."""
import rdkit
from rdkit import Chem
import argparse
import json
import logging
import os
import torch
import pickle
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
from collections import OrderedDict
from paccmann_predictor.utils.utils import get_device
import pandas as pd
from cytotox.models import MODEL_FACTORY
from pytoda.smiles.smiles_language import SMILESLanguage
from pytoda.smiles.transforms import SMILESToTokenIndexes
from cytotox.utils.utils import disable_rdkit_logging
from .drug_evaluator import DrugEvaluator
from pytoda.transforms import Transform
from pytoda.smiles.transforms import LeftPadding

# tox21_model_path = os.path.join(
#     os.path.expanduser('~'),
#     'Box/Molecular_SysBio/data/cytotoxicity/models/Tox21/raw_aug_MCA_5'
# )

# smiles_language_path = os.path.join(
#     os.path.expanduser('~'),
#     'Box/Molecular_SysBio/data/cytotoxicity/smiles/smiles_language_chembl_gdsc_ccle_tox21_zinc.pkl'
# )



class Tox21(DrugEvaluator):
    """
    Tox21 evaluation class.
    Inherits from DrugEvaluator and evaluates the Tox21 score of a SMILES.
    """

    def __init__(self):

        super(Tox21, self).__init__()

    def __call__(self, mol):
        """
        Returns the Tox21 of a SMILES string or a RdKit molecule.
        """
        # Check if molecule is valid
        # Error handling.
        if type(mol) == rdkit.Chem.rdchem.Mol:
            pass
        elif type(mol) == str:
            mol = Chem.MolFromSmiles(mol, sanitize=False)
            if mol is None:
                raise ValueError("Invalid SMILES string.")
        else:
            raise TypeError("Input must be from {str, rdkit.Chem.rdchem.Mol}")
        
        return self.tox21_score(mol)
    

    def tox21_score(self, mol)
        # TODO: load model
        
        
        with open(os.path.join(tox21_model_path, 'model_params.json')) as f:
            tox21_params = json.load(f)
        mol = MODEL_FACTORY['mca'](tox21_params)
        
        #TODO: Compose(transforms)
        
        
        # Test the compound
        smiles_t = LeftPadding(Chem.MolFromSmiles(mol),pad_len=300)
        pred_tox21_per_task, pred_dict = tox21_predictor(smiles_t)
        pred_tox21_average = sum(pred_tox21_per_task)/12
        tox21_score = pred_tox21_average

        return tox21_score
