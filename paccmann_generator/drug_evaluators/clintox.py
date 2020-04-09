#%%
"""ClinTox evaluator."""
from .drug_evaluator import DrugEvaluator


class ClinTox(DrugEvaluator):
    """
    ClinTox evaluation class.
    Inherits from DrugEvaluator and evaluates the clinically observed toxicity.
    2 Classes:
        First one is probability to receive FDA approval.
        Second is probability of failure in clinical stage.
    """

    def __init__(self, model_path, reward_type='thresholded'):
        """

        Arguments:
            model_path {string} -- Path to the pretrained model
        
        Keyword Arguments:
            reward_type {string} -- From {'thresholded', 'raw'}. If 'raw' the
                average of the raw predictions is used as reward.
                If `thresholded`, reward is 1 iff probability to get approval
                is > 0.5 and probability of clinical failure is < 0.5.
        """

        super(ClinTox, self).__init__()
        self.load_mca(model_path)

        self.set_reward_fn(reward_type)

    def set_reward_fn(self, reward_type):
        self.reward_type = reward_type
        if reward_type == 'thresholded':
            self.reward_fn = lambda x: (
                1. if x[:, 0] > .5 and x[:, 1] < .5 else 0.
            )
        elif reward_type == 'raw':
            # Average probabilities
            self.reward_fn = lambda x: float((x[0] + 1 - x[1]) / 2)
        else:
            raise ValueError(f'Unknown reward_type given: {reward_type}')

    def __call__(self, smiles):
        """
        Forward pass through the model.

        Arguments:
            smiles {str} -- SMILES of molecule
        Returns:
            float -- Averaged  predictions from the model

        TODO: Should be able to understand iterables
        """
        # Error handling.
        if not type(smiles) == str:
            raise TypeError(f'Input must be String, not :{type(smiles)}')

        smiles_tensor = self.preprocess_smiles(smiles)
        return self.clintox_score(smiles_tensor)

    def clintox_score(self, smiles_tensor):
        """
        Forward pass through the model.

        Arguments:
            smiles_tensor {str} -- SMILES

        Returns:
            float -- Averaged  predictions from the model
        """

        # Test the compound
        predictions, _ = self.model(smiles_tensor)
        # To allow accessing the raw predictions from outside
        self.predictions = predictions[0, :]

        return self.reward_fn(self.predictions)
