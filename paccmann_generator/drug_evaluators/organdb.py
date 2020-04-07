"""OrganDB evaluator."""
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


class OrganDB(DrugEvaluator):
    """
    OrganDB evaluation class.
    Inherits from DrugEvaluator and evaluates the OrganDB score of a SMILES.
    Organs can be:
        'Adrenal Gland', 'Bone Marrow', 'Brain', 'Eye',
        'Heart', 'Kidney', 'Liver', 'Lung', 'Lymph Node',
        'Mammary Gland', 'Pancreas', 'Pituitary Gland',
        'Spleen', 'Stomach', 'Testes', 'Thymus',
        'Thyroid Gland', 'Urinary Bladder', 'Uterus', 'Ovary'
    """

    def __init__(self):

        super(OrganDB, self).__init__()

    def __call__(self, mol, organ):
        """
        Returns the OrganDB of a SMILES string or a RdKit molecule.
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
        
        return self.organdb_score(mol, *organ)

        
    def task_specificity(self, pred_organdb_per_task, organ):
        task_names = [
            'CHR:Adrenal Gland', 'CHR:Bone Marrow', 'CHR:Brain', 'CHR:Eye',
            'CHR:Heart', 'CHR:Kidney', 'CHR:Liver', 'CHR:Lung', 'CHR:Lymph Node',
            'CHR:Mammary Gland', 'CHR:Pancreas', 'CHR:Pituitary Gland',
            'CHR:Spleen', 'CHR:Stomach', 'CHR:Testes', 'CHR:Thymus',
            'CHR:Thyroid Gland', 'CHR:Urinary Bladder', 'CHR:Uterus', 'MGR:Brain',
            'MGR:Kidney', 'MGR:Ovary', 'MGR:Testes', 'SUB:Adrenal Gland',
            'SUB:Bone Marrow', 'SUB:Brain', 'SUB:Heart', 'SUB:Kidney', 'SUB:Liver',
            'SUB:Lung', 'SUB:Spleen', 'SUB:Stomach', 'SUB:Testes', 'SUB:Thymus',
            'SUB:Thyroid Gland'
        ]
        chronics = []
        for ind,i in enumerate(task_names):
            if organ in i:
                chronics.append(pred_organdb_per_task[ind])
        average_over_chronics = sum(chronics)/len(chronics)
        
        return average_over_chronics
    
    def organdb_score(self, mol, organ):
        # model.load()
        # Test the compound
        smiles_t = LeftPadding(Chem.MolFromSmiles(mol),pad_len=300)
        pred_organdb_per_task, pred_dict = organdb_predictor(smiles_t)
        #pred_organdb_average = sum(pred_tox21_per_task)/35
        average_over_chronics = task_specificity(pred_organdb_per_task,organ)
        
        return average_over_chronics
