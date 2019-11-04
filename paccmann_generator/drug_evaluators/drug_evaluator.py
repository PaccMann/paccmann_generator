"""Drug evaluator."""


class DrugEvaluator:
    """
    Abstract definition of DrugEvaluator class.
    This scaffold is supposed  to be extended by specific
    drug evaluation metrics.
    """

    def __call__(self, smiles):

        raise NotImplementedError
