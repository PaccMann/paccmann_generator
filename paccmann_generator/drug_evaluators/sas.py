"""SAS evaluator."""
import gzip
import math
import pickle
import rdkit
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from .drug_evaluator import DrugEvaluator
from ..io import get_file


class SAS(DrugEvaluator):
    """
    SAS evaluation class.
    Inherits from DrugEvaluator and computes the SAS (synthetic assebility
    score) for a molecule.
    """

    fpscore_file_name = 'fpscores.pkl.gz'
    fpscore_url = (
        'https://github.com/rdkit/rdkit/raw/'
        '4081cb51e5337230240fb9073b6ca4ef903f94a5/Contrib/SA_Score/' +
        fpscore_file_name
    )

    def __init__(self):

        super(SAS, self).__init__()

        self.fpscores_path = get_file(self.fpscore_file_name, self.fpscore_url)

        self._fscores = self.get_fscores()

    def __call__(self, mol):
        """
        Returns the SAS of a SMILES string or a RdKit molecule.
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

        return self.sa_score(mol)

    def get_fscores(self):
        _fscores = pickle.load(gzip.open(self.fpscores_path))
        outDict = {}
        for i in _fscores:
            for j in range(1, len(i)):
                outDict[i[j]] = float(i[0])
        _fscores = outDict
        return _fscores

    def numBridgeheadsAndSpiro(self, mol, ri=None):
        nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        return nBridgehead, nSpiro

    def sa_score(self, mol):
        # fragment score
        fp = rdMolDescriptors.GetMorganFingerprint(mol, 2)
        fps = fp.GetNonzeroElements()
        score1 = 0.
        nf = 0
        for bitId, v in fps.items():
            nf += v
            sfp = bitId
            score1 += self._fscores.get(sfp, -4) * v
        score1 /= nf

        # features score
        nAtoms = mol.GetNumAtoms()
        nChiralCenters = len(
            Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        )
        ri = mol.GetRingInfo()
        nBridgeheads, nSpiro = self.numBridgeheadsAndSpiro(mol, ri)
        nMacrocycles = 0
        for x in ri.AtomRings():
            if len(x) > 8:
                nMacrocycles += 1

        sizePenalty = nAtoms**1.005 - nAtoms
        stereoPenalty = math.log10(nChiralCenters + 1)
        spiroPenalty = math.log10(nSpiro + 1)
        bridgePenalty = math.log10(nBridgeheads + 1)
        macrocyclePenalty = 0.
        # ---------------------------------------
        # This differs from the paper, which defines:
        #  macrocyclePenalty = math.log10(nMacrocycles+1)
        # This form generates better results when 2 or more macrocycles are present
        if nMacrocycles > 0:
            macrocyclePenalty = math.log10(2)

        score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

        # correction for the fingerprint density
        # not in the original publication, added in version 1.1
        # to make highly symmetrical molecules easier to synthetise
        score3 = 0.
        if nAtoms > len(fps):
            score3 = math.log(float(nAtoms) / len(fps)) * .5

        sascore = score1 + score2 + score3

        # need to transform "raw" value into scale between 1 and 10
        min = -4.0
        max = 2.5
        sascore = 11. - (sascore - min + 1) / (max - min) * 9.
        # smooth the 10-end
        if sascore > 8.:
            sascore = 8. + math.log(sascore + 1. - 9.)
        if sascore > 10.:
            sascore = 10.0
        elif sascore < 1.:
            sascore = 1.0

        return sascore
