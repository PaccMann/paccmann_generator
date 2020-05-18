"""Computes penalized logP evaluator."""
import rdkit
from rdkit import Chem
from .drug_evaluator import DrugEvaluator
from .sas import SAS


class PenalizedLogP(DrugEvaluator):
    """
    PenalizedLogP evaluation class.
    Inherits from DrugEvaluator and evaluates PenalizedLogP score of a SMILES.
    """

    def __init__(self):

        super(PenalizedLogP, self).__init__()
        self.logp = lambda mol: Chem.Crippen.MolLogP(mol)
        self.sas = SAS()

    def __call__(self, mol):
        """
        Returns the PenalizedLogP of a SMILES string or a RdKit molecule.
        """

        # Error handling.
        if type(mol) == rdkit.Chem.rdchem.Mol:
            pass
        elif type(mol) == str:
            mol = Chem.MolFromSmiles(mol, sanitize=False)
            if mol is None:
                raise ValueError("Invalid SMILES string.")
        else:
            raise TypeError("Input must be from {str, rdkit.Chem.rdchem.Mol}")

        try:
            return self.logp(mol) + self.get_num_rings_6(mol) + self.sas(mol)
        # Catch atom valence exception raised by CalcCrippenDescriptor
        except Exception:
            return 0.

    @staticmethod
    def get_num_rings_6(mol):
        r = mol.GetRingInfo()
        return len([x for x in r.AtomRings() if len(x) > 6])
