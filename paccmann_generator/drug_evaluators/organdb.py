"""OrganDB evaluator."""
import torch
from .drug_evaluator import DrugEvaluator

TASK_NAMES = [
    'CHR:Adrenal Gland', 'CHR:Bone Marrow', 'CHR:Brain', 'CHR:Eye',
    'CHR:Heart', 'CHR:Kidney', 'CHR:Liver', 'CHR:Lung', 'CHR:Lymph Node',
    'CHR:Mammary Gland', 'CHR:Pancreas', 'CHR:Pituitary Gland', 'CHR:Spleen',
    'CHR:Stomach', 'CHR:Testes', 'CHR:Thymus', 'CHR:Thyroid Gland',
    'CHR:Urinary Bladder', 'CHR:Uterus', 'MGR:Brain', 'MGR:Kidney',
    'MGR:Ovary', 'MGR:Testes', 'SUB:Adrenal Gland', 'SUB:Bone Marrow',
    'SUB:Brain', 'SUB:Heart', 'SUB:Kidney', 'SUB:Liver', 'SUB:Lung',
    'SUB:Spleen', 'SUB:Stomach', 'SUB:Testes', 'SUB:Thymus',
    'SUB:Thyroid Gland'
]

SITES = [
    'Adrenal Gland', 'Bone Marrow', 'Brain', 'Eye', 'Heart', 'Kidney', 'Liver',
    'Lung', 'Lymph Node', 'Mammary Gland', 'Pancreas', 'Pituitary Gland',
    'Spleen', 'Stomach', 'Testes', 'Thymus', 'Thyroid Gland',
    'Urinary Bladder', 'Uterus', 'Ovary'
]
TOXICITIES = {
    'chronic': ['CHR'],
    'subchronic': ['SUB'],
    'multigenerational': ['MGR'],
    'all': ['CHR', 'SUB', 'MGR']
}


class OrganDB(DrugEvaluator):
    """
    OrganDB evaluation class.
    Inherits from DrugEvaluator and evaluates the OrganDB score of a SMILES.
    Organs can be:
        'Adrenal Gland', 'Bone Marrow', 'Brain', 'Eye',
        'Heart', 'Kidney', 'Liver', 'Lung', 'Lymph Node',
        'Mammary Gland', 'Pancreas', 'Pituitary Gland',
        'Spleen', 'Stomach', 'Testes', 'Thymus',
        'Thyroid Gland', 'Urinary Bladder', 'Uterus', 'Ovary'

    """

    def __init__(
        self,
        model_path: str,
        site: str,
        toxicity_type: str = 'all',
        reward_type: str = 'thresholded'
    ):
        """
        Arguments:
            model_path (string): Path to the pretrained model
            site (string):  Name of organ of interest

        Keyword Arguments:
            toxicity_type (string): Type of toxicity from
                {'all', 'chronic', 'subcronic', 'multigenerational'}. Defaults
                to 'all'.
            reward_type {string} -- From {'thresholded', 'raw'}. If 'raw' the
                (inverted) average of the raw predictions is used as reward.
                If `thresholded`, reward is 1 iff all predictions are < 0.5.
        """

        super(OrganDB, self).__init__()
        self.load_mca(model_path)
        self.set_reward_fn(site, toxicity_type, reward_type)

    def set_reward_fn(self, site: str, toxicity_type: str, reward_type: str):
        """
        Updates the reward function.
        Arguments:
            toxicity_type (string): Type of toxicity from
                {'all', 'chronic', 'subcronic', 'multigenerational'}. Defaults
                to 'all'.
            reward_type (string): From {'thresholded', 'raw'}. If 'raw' the
                (inverted) average of the raw predictions is used as reward.
                If `thresholded`, reward is 1 iff all predictions are < 0.5.
        """

        # Error handling
        site = site.strip().lower()
        if not any(list(map(lambda s: site in s.strip().lower(), SITES))):
            raise ValueError(f'Unknown site: ({site}). Chose from: {SITES}')

        toxicity_type = toxicity_type.strip().lower()
        if toxicity_type not in TOXICITIES.keys():
            raise ValueError(
                f'Unknown toxicity type: {toxicity_type}. '
                f'Chose from: {TOXICITIES.keys()}'
            )

        # Select right classes
        class_indices = [
            i for i, x in enumerate(
                list(
                    map(
                        lambda task: site.lower() in task.lower() and any(
                            list(
                                map(
                                    lambda t: t.lower() in task.lower(),
                                    TOXICITIES[toxicity_type]
                                )
                            )
                        ), TASK_NAMES
                    )
                )
            ) if x
        ]
        if len(class_indices) == 0:
            raise ValueError(
                f'Model cannot perform predictions for organ: {site} and '
                f'toxicity type: {toxicity_type}.'
            )
        self.toxicity_type = toxicity_type
        self.site = site
        self.class_indices = class_indices

        # Class names
        self.classes = [TASK_NAMES[c] for c in class_indices]

        if reward_type == 'thresholded':
            # If any assay was positive, no reward is given
            self.reward_fn = lambda yhat, classes: 0. if any(
                yhat[classes] > 0.5
            ) else 1.
        elif reward_type == 'raw':
            # Average probabilities and invert to get reward
            self.reward_fn = lambda yhat, classes: 1. - float(
                yhat[classes].mean()
            )
        else:
            raise ValueError(f'Unknown reward_type given: {reward_type}')

        self.reward_type = reward_type

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
        return self.organdb_score(smiles_tensor)

    def organdb_score(self, smiles_tensor: torch.Tensor) -> float:
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

        return self.reward_fn(self.predictions, self.class_indices)
