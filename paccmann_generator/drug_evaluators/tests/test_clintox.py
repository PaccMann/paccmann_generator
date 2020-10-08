"""Testing ClinTox model."""
import unittest
import os
from paccmann_generator.drug_evaluators import ClinTox
"""
class TestClinTox(unittest.TestCase):
    "Testing OrganDB model "

    def test_set_reward_fn(self) -> None:
        "Test set_reward_fn."

        path = os.path.join(
             'data', 'models', 'ClinToxMulti'
        )

        for reward in ['thresholded', 'raw']:
            model = ClinTox(path, reward)
            self.assertGreaterEqual(model('CCO'), 0)


if __name__ == '__main__':
    unittest.main()
"""