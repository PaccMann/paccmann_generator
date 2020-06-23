"""Testing ClinTox model."""
import unittest
from paccmann_generator.drug_evaluators import AromaticRing


class TestAromaticRing(unittest.TestCase):
    """Testing OrganDB model """

    def test_set_reward_fn(self) -> None:
        """Test set_reward_fn."""

        f = AromaticRing()

        for s, gt in zip(
            ['Cc1ccc(cc1S(C)(=O)=O)C(=O)NCc1cccc(OC)c1', 'OC(=O)C'], [1., 0.]
        ):
            print(s, f(s))
            self.assertEqual(f(s), gt)


if __name__ == '__main__':
    unittest.main()
