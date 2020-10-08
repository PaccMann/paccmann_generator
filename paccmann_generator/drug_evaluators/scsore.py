"""SCSore evaluator."""
import gzip
import json
import math
import pickle

import numpy as np
import rdkit
import rdkit.Chem.AllChem as AllChem
import six
from rdkit import Chem

from ..io import get_file
from .drug_evaluator import DrugEvaluator


class SCScore(DrugEvaluator):
    """
    SCScore evaluation class.
    Inherits from DrugEvaluator and computes the SCScore (Synthetic Complexity
    SCORE) for a molecule.

    Based upon:
        Coley, Connor W., et al. "SCScore: Synthetic complexity learned from a
            reaction corpus." Journal of chemical information and modeling 58.2
            (2018): 252-261.

    Code adopted from: https://github.com/connorcoley/scscore

    """

    scscore_filename = 'model.ckpt-10654.as_numpy.json.gz?raw=true'
    scscore_url = (
        'https://github.com/connorcoley/scscore/blob/master/models/'
        'full_reaxys_model_1024bool/' + scscore_filename
    )

    def __init__(self, score_scale=5.0, fp_len=1024, fp_rad=2):

        super(SCScore, self).__init__()

        self.vars = []
        self.score_scale = score_scale
        self.fp_len = fp_len
        self.fp_rad = fp_rad
        self.sigmoid = lambda x: 1 / (1 + math.exp(x))

        self.scscore_path = get_file(self.scscore_filename, self.scscore_url)
        self.restore()

        self._restored = True

    def __call__(self, mol):

        # Error handling.
        if not self._restored:
            raise ValueError('Must restore model weights!')

        if type(mol) == rdkit.Chem.rdchem.Mol:
            smiles = Chem.MolToSmiles(mol, canonical=True)
        elif type(mol) == str:
            smiles = mol
            molecule = Chem.MolFromSmiles(smiles, sanitize=False)
            if molecule is None:
                raise ValueError("Invalid SMILES string.")
        else:
            raise TypeError("Input must be from {str, rdkit.Chem.rdchem.Mol}")

        fp = np.array((self.smi_to_fp(smiles)), dtype=np.float32)

        cur_score = self._compute_scscore(fp) if sum(fp) != 0 else 0.

        return cur_score

    def _compute_scscore(self, x):

        for i in range(0, len(self.vars), 2):
            last_layer = (i == len(self.vars) - 2)
            W = self.vars[i]
            b = self.vars[i + 1]
            x = np.matmul(x, W) + b
            if not last_layer:
                x = x * (x > 0)  # ReLU
        x = 1 + (self.score_scale - 1) * self.sigmoid(x)
        return x

    def restore(self):
        self._load_vars(self.scscore_path)

        if 'uint8' in self.scscore_path or 'counts' in self.scscore_path:

            def mol_to_fp(self, mol):
                if mol is None:
                    return np.array((self.fp_len, ), dtype=np.uint8)
                fp = AllChem.GetMorganFingerprint(
                    mol, self.fp_rad, useChirality=True
                )  # uitnsparsevect
                fp_folded = np.zeros((self.fp_len, ), dtype=np.uint8)
                for k, v in six.iteritems(fp.GetNonzeroElements()):
                    fp_folded[k % self.fp_len] += v
                return np.array(fp_folded)
        else:

            def mol_to_fp(self, mol):
                if mol is None:
                    return np.zeros((self.fp_len, ), dtype=np.float32)
                return np.array(
                    AllChem.GetMorganFingerprintAsBitVect(
                        mol, self.fp_rad, nBits=self.fp_len, useChirality=True
                    ),
                    dtype=np.bool
                )

        self.mol_to_fp = mol_to_fp
        self._restored = True
        return self

    def smi_to_fp(self, smi):

        if not smi:
            return np.zeros((self.fp_len, ), dtype=np.float32)

        return self.mol_to_fp(self, Chem.MolFromSmiles(smi))

    def _load_vars(self, weight_path):

        if 'pickle' in weight_path.split('/')[-1]:
            with open(weight_path, 'rb') as fid:
                self.vars = pickle.load(fid)
                self.vars = [x.tolist() for x in self.vars]

        elif 'json.gz' in weight_path.split('/')[-1]:
            with gzip.GzipFile(weight_path, 'r') as fin:  # 4. gzip
                json_bytes = fin.read()  # 3. bytes (i.e. UTF-8)
                json_str = json_bytes.decode('utf-8')  # 2. string (i.e. JSON)
                self.vars = json.loads(json_str)
                self.vars = [np.array(x) for x in self.vars]

        else:
            raise NotImplementedError('Provide .pickle or .json.gz file')
