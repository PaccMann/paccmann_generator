"""Aromatic Ring evaluator."""
import logging

import rdkit
from rdkit import Chem

from .drug_evaluator import DrugEvaluator

logger = logging.getLogger(__name__)


class AromaticRing(DrugEvaluator):
    """
    Evaluation class that assesses whether a molecule has an aromatic ring.

    99% of drugs have at least one aromatic ring:

        Roughley, Stephen D., and Allan M. Jordan. "The medicinal chemist’s
        toolbox: an analysis of reactions used in the pursuit of drug
        candidates." Journal of medicinal chemistry 54.10 (2011): 3451-3479.

    """

    def __init__(self):

        super(AromaticRing, self).__init__()

    def __call__(self, mol):
        """
        Returns 1 if mol has at least one aromatic ring and 0 otherwise.

        Args:
            mol - Union[str, rdkit.Chem.rdchem.Mol]: SMILES or RdKit molecule.
        Returns:
            float - 1. if aromatic ring was found, 0 else.
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

        try:
            has_ring = False
            for ring in mol.GetRingInfo().AtomRings():
                if any([mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring]):
                    has_ring = True
                if has_ring:
                    break
            return 1. if has_ring else 0.

        except Exception:
            logger.warn(f'Error in computing ring information for {mol}')
            return 0.
