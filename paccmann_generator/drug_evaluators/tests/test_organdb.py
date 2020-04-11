"""Testing OrganDB model."""
import unittest
import torch
import numpy as np
from paccmann_generator.drug_evaluators import OrganDB


# To bypass constructor (model loading)
class FakeOrganDB(OrganDB):

    def __init__(self):
        pass


class TestOrganDB(unittest.TestCase):
    """Testing OrganDB model """

    def test_set_reward_fn(self) -> None:
        """Test set_reward_fn."""

        sites = ['lUnG  ', 'Brain']
        toxicity_types = [' ChroniC ', 'ALL']
        reward_types = ['thresholded', 'raw']
        gt_classes = [
            ['CHR:Lung'], ['CHR:Lung'], ['CHR:Lung', 'SUB:Lung'],
            ['CHR:Lung', 'SUB:Lung'], ['CHR:Brain'], ['CHR:Brain'],
            ['CHR:Brain', 'MGR:Brain', 'SUB:Brain'],
            ['CHR:Brain', 'MGR:Brain', 'SUB:Brain']
        ]
        dummy_predictions = torch.arange(35) / 35.
        gt_rewards = [
            1., 1 - (7 / 35), 0., 1 - (((7 / 35) + (29 / 35)) / 2), 1.,
            1 - (2 / 35), 0., 1 - (((2 / 35) + (19 / 35) + (25 / 35)) / 3)
        ]

        ind = -1
        for site in sites:
            for toxicity_type in toxicity_types:
                for reward_type in reward_types:
                    ind += 1
                    o = FakeOrganDB()
                    o.set_reward_fn(
                        site=site,
                        toxicity_type=toxicity_type,
                        reward_type=reward_type
                    )
                    self.assertEqual(o.classes, gt_classes[ind])
                    self.assertEqual(o.site, site.lower().strip())
                    reward = o.reward_fn(dummy_predictions, o.class_indices)
                    self.assertTrue(np.allclose(reward, gt_rewards[ind]))


if __name__ == '__main__':
    unittest.main()
