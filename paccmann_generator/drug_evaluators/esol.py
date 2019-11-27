"""ESOL evaluator."""
from collections import namedtuple
import rdkit
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski
from .drug_evaluator import DrugEvaluator


class ESOL(DrugEvaluator):
    """
    ESOL evaluation class.
    Inherits from DrugEvaluator and estimates the solubility score of a 
    molecule.
    """

    def __init__(self):

        super(ESOL, self).__init__()
        self.aromatic_query = Chem.MolFromSmarts("a")
        self.Descriptor = namedtuple("Descriptor", "mw logp rotors ap")

    def __call__(self, mol):
        """
        Returns the ESOL of a SMILES string or a RdKit molecule.
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

        return self.calc_esol(mol)

    def calc_ap(self, mol):
        """
        Calculate aromatic proportion #aromatic atoms/#atoms total
        :param mol: input molecule
        :return: aromatic proportion
        """
        matches = mol.GetSubstructMatches(self.aromatic_query)
        return len(matches) / mol.GetNumAtoms()

    def calc_esol_descriptors(self, mol):
        """
        Calcuate mw,logp,rotors and aromatic proportion (ap)
        :param mol: input molecule
        :return: named tuple with descriptor values
        """
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        rotors = Lipinski.NumRotatableBonds(mol)
        ap = self.calc_ap(mol)
        return self.Descriptor(mw=mw, logp=logp, rotors=rotors, ap=ap)

    def calc_esol_orig(self, mol):
        """
        Original parameters from the Delaney paper, just here for comparison
        :param mol: input molecule
        :return: predicted solubility
        """
        # just here as a reference don't use this!
        intercept = 0.16
        coef = {"logp": -0.63, "mw": -0.0062, "rotors": 0.066, "ap": -0.74}
        desc = self.calc_esol_descriptors(mol)
        esol = (
            intercept + coef["logp"] * desc.logp + coef["mw"] * desc.mw +
            coef["rotors"] * desc.rotors + coef["ap"] * desc.ap
        )
        return esol

    def calc_esol(self, mol):
        """
        Calculate ESOL based on descriptors in the Delaney paper, coefficients 
        refit for the RDKit using the routine refit_esol below.
        :param mol: input molecule
        :return: predicted solubility
        """
        intercept = 0.26121066137801696
        coef = {
            'mw': -0.0066138847738667125,
            'logp': -0.7416739523408995,
            'rotors': 0.003451545565957996,
            'ap': -0.42624840441316975
        }
        desc = self.calc_esol_descriptors(mol)
        esol = (
            intercept + coef["logp"] * desc.logp + coef["mw"] * desc.mw +
            coef["rotors"] * desc.rotors + coef["ap"] * desc.ap
        )
        return esol
