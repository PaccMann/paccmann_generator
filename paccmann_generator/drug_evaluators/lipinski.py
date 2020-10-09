"""Lipinski rule of five evaluator."""
from collections import namedtuple
import rdkit
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors
from rdkit.Chem import Lipinski as Lipi
from .drug_evaluator import DrugEvaluator


class Lipinski(DrugEvaluator):
    """
    Lipinski evaluation class.
    Inherits from DrugEvaluator and evaluates the four Lipinski rules

    Implementation taken from: https://gist.github.com/strets123/fdc4db6d450b66345f46
    """

    def __init__(self):

        super(Lipinski, self).__init__()
        self.aromatic_query = Chem.MolFromSmarts("a")
        self.Descriptor = namedtuple("Descriptor", "mw logp rotors ap")

    def __call__(self, mol):
        """
        Args:
            - mol {rdkit.Chem.rdchem.Mol, str}

        Returns:     a tuple consisting of:
            - a boolean indicating whether the molecule passed Lipinski test
            - a dictionary giving the values of the Lipinski check.
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

        return self.calc_lipinski(mol)

    def calc_lipinski(self, mol):
        """
        Returns:     a tuple consisting of:
            - a boolean indicating whether the molecule passed Lipinski test
            - a dictionary giving the values of the Lipinski check.

        NOTE:   Lipinski's rules are:
            - Hydrogen bond donors <= 5
            - Hydrogen bond acceptors <= 10
            - Molecular weight < 500 daltons
            - logP < 5
        """

        num_hdonors = Lipi.NumHDonors(mol)
        num_hacceptors = Lipi.NumHAcceptors(mol)
        mol_weight = Descriptors.MolWt(mol)
        mol_logp = round(Crippen.MolLogP(mol), 4)

        pass_num_hdonors = num_hdonors <= 5
        pass_num_hacceptors = num_hacceptors <= 10
        pass_mol_weight = mol_weight < 500
        pass_mol_logp = mol_logp < 5

        return (
            (
                pass_num_hdonors and pass_num_hacceptors and pass_mol_weight
                and pass_mol_logp
            ), {
                'hydrogen_bond_donors': num_hdonors,
                'hydrogen_bond_acceptors': num_hacceptors,
                'molecular_weight': mol_weight,
                'logp': mol_logp,
                'test_results':
                    {
                        'hydrogen_bond_donors': pass_num_hdonors,
                        'hydrogen_bond_acceptors': pass_num_hacceptors,
                        'molecular_weight': pass_mol_weight,
                        'logp': pass_mol_logp
                    }
            }
        )
