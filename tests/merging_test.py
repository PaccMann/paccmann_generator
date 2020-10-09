import pytest
import numpy as np
import pandas as pd
import torch
from paccmann_generator.reinforce_proteins_omics import ReinforceProteinOmics

def test_together_concat():
    a = torch.tensor([[[1,2,3], [1,2,3]]], dtype=torch.float)
    b = torch.tensor([[[4,5,6], [4,5,6]]], dtype=torch.float)
    res = ReinforceProteinOmics.together_concat([1,2,3],b,a)
    assert torch.is_tensor(res)
    assert torch.equal(res, torch.tensor([[[1,2,3,4,5,6], [1,2,3,4,5,6]]], dtype=torch.float))

def test_together_mean():
    a = torch.tensor([[[1,2,3], [1,2,3]]], dtype=torch.float)
    b = torch.tensor([[[4,5,6], [4,5,6]]], dtype=torch.float)
    res = ReinforceProteinOmics.together_mean([1,2,3],b,a)
    assert torch.is_tensor(res)
    assert torch.equal(res, torch.tensor([[2.5,3.5,4.5], [2.5,3.5,4.5]], dtype=torch.float))

test_together_mean()
test_together_concat()
print("all test terminated successfully!")