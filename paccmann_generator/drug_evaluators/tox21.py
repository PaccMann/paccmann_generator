#%%
"""Tox21 evaluator."""
from .drug_evaluator import DrugEvaluator


class Tox21(DrugEvaluator):
    """
    Tox21 evaluation class.
    Inherits from DrugEvaluator and evaluates the Tox21 score of a SMILES.
    """

    def __init__(self, model_path, reward_type='thresholded'):
        """

        Arguments:
            model_path {string} -- Path to the pretrained model

        Keyword Arguments:
            reward_type {string} -- From {'thresholded', 'raw'}. If 'raw' the
                (inverted) average of the raw predictions is used as reward.
                If `thresholded`, reward is 1 iff all predictions are < 0.5.
        """

        super(Tox21, self).__init__()
        self.load_mca(model_path)

    def __call__(self, smiles):
        """
        Forward pass through the model.

        Arguments:
            smiles {str} -- SMILES of molecule
        Returns:
            float -- Averaged tox21 predictions from the model

        TODO: Should be able to understand iterables
        """
        # Error handling.
        if not type(smiles) == str:
            raise TypeError(f'Input must be String, not :{type(smiles)}')

        smiles_tensor = self.preprocess_smiles(smiles)
        return self.tox21_score(smiles_tensor)

    def tox21_score(self, smiles_tensor):
        """
        Forward pass through the model.

        Arguments:
            smiles_tensor {str} -- SMILES

        Returns:
            float -- [description]
        """

        # Test the compound
        predictions, _ = self.model(smiles_tensor)
        # To allow accessing the raw predictions from outside
        self.predictions = predictions[0, :]

        # If any assay was positive, no reward is given
        return 0. if any(self.predictions > 0.5) else 1.
