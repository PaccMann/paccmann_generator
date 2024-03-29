"""Tox21 evaluator."""
from .drug_evaluator import DrugEvaluator
import torch


class Tox21(DrugEvaluator):
    """
    Tox21 evaluation class.
    Inherits from DrugEvaluator and evaluates the Tox21 score of a SMILES.
    """

    def __init__(self, model_path: str, reward_type: str = 'thresholded'):
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

        self.set_reward_fn(reward_type)

    def set_reward_fn(self, reward_type: str):
        self.reward_type = reward_type
        if reward_type == 'thresholded':
            # If any assay was positive, no reward is given
            self.reward_fn = lambda x: 0.0 if any(x > 0.5) else 1.0
        elif reward_type == 'raw':
            # Average probabilities and invert to get reward
            self.reward_fn = lambda x: 1.0 - float(x.mean())
        else:
            raise ValueError(f'Unknown reward_type given: {reward_type}')

    def __call__(self, smiles: str) -> float:
        """
        Forward pass through the model.

        Arguments:
            smiles {str} -- SMILES of molecule
        Returns:
            float -- Reward used for the generator (high reward = low toxicity)

        TODO: Should be able to understand iterables
        """
        # Error handling.
        if not type(smiles) == str:
            raise TypeError(f'Input must be String, not :{type(smiles)}')

        smiles_tensor = self.preprocess_smiles(smiles)
        return self.tox21_score(smiles_tensor)

    def tox21_score(self, smiles_tensor: torch.Tensor) -> float:
        """
        Forward pass through the model.

        Arguments:
            smiles_tensor {torch.Tensor} -- Tensor of shape 2 x SMILES_tokens

        Returns:
            float -- Reward
        """

        # Test the compound
        predictions, _ = self.model(smiles_tensor)
        # To allow accessing the raw predictions from outside
        self.predictions = predictions[0, :]

        return self.reward_fn(self.predictions)
