"""QED evaluator."""
import rdkit
from rdkit import Chem
from rdkit.Chem.Descriptors import qed
from .drug_evaluator import DrugEvaluator


class QED(DrugEvaluator):
    """
    QED evaluation class.
    Inherits from DrugEvaluator and evaluates the QED score of a SMILES.
    """

    def __init__(self):

        super(QED, self).__init__()

    def __call__(self, mol):
        """
        Returns the QED of a SMILES string or a RdKit molecule.
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

        return qed(mol)
