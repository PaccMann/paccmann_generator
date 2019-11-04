"""Molecular weight evaluator."""
import rdkit
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt
from .drug_evaluator import DrugEvaluator


class MolecularWeight(DrugEvaluator):
    """
    Molecula weight evaluation class.
    Inherits from DrugEvaluator and evaluates the molecular weight of a SMILES.
    """

    def __init__(self):

        super(MolecularWeight, self).__init__()

    def __call__(self, mol):
        """
        Returns the QED of a SMILES string or a RdKit molecule.
        """

        # Error handling.
        if type(mol) == rdkit.Chem.rdchem.Mol:
            pass
        elif type(mol) == str:
            mol = Chem.MolFromSmiles(mol, sanitize=True)
            if mol is None:
                raise ValueError("Invalid SMILES string.")
        else:
            raise TypeError("Input must be from {str, rdkit.Chem.rdchem.Mol}")

        return MolWt(mol)
