import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from itertools import product

trans_data_path = '/home/tol/data/gsdc_omics_latent.csv'
proteo_data_path = '/home/tol/data/gsdc_proteins_latent.csv'

transcriptomics = pd.read_csv(trans_data_path)
#first remove test cell_line
test_cell_line = transcriptomics.cell_line[0]
transcriptomics_train = transcriptomics[~(transcriptomics.cell_line == test_cell_line)]
print("test cell line:", test_cell_line,"; rest:\n",transcriptomics_train.cell_line)
proteomics = pd.read_csv(proteo_data_path)

t_latent_columns = transcriptomics_train.columns[list(range(-128, 0, 1))]
t_latent_code = transcriptomics_train[t_latent_columns]

p_latent_columns = proteomics.columns[list(range(-128, 0, 1))]
p_latent_code = proteomics[p_latent_columns]

#split data sets
t_train, t_valid_test = train_test_split(t_latent_code, test_size=2)
t_valid, t_test = train_test_split(t_valid_test, test_size=1)
print(t_train.shape, t_valid.shape, t_test.shape)

p_train, p_valid_test = train_test_split(p_latent_code, test_size=30)
p_valid, p_test = train_test_split(p_valid_test, test_size=10)
print(p_train.shape, p_valid.shape, p_test.shape)

tp_train = list(product(t_train.values, p_train.values))
len_tp_train = len(tp_train)
#assert len_tp_train == 398182
batch_train_idxs = torch.arange(len_tp_train).unsqueeze(1).repeat(1, 2)
tp_train_perms = torch.stack(list(map(torch.randperm, [2] * len_tp_train)))
tp_train = torch.as_tensor(tp_train)[batch_train_idxs, tp_train_perms, :]

tp_valid = list(product(t_valid.values, p_valid.values))
len_tp_valid = len(tp_valid)
assert len_tp_valid == 20
batch_valid_idxs = torch.arange(len_tp_valid).unsqueeze(1).repeat(1, 2)
tp_valid_perms = torch.stack(list(map(torch.randperm, [2] * len_tp_valid)))
tp_valid = torch.as_tensor(tp_valid)[batch_valid_idxs, tp_valid_perms, :]

tp_test = list(product(t_test.values, p_test.values))
len_tp_test = len(tp_test)
assert len_tp_test == 10
batch_test_idxs = torch.arange(len_tp_test).unsqueeze(1).repeat(1, 2)
tp_test_perms = torch.stack(list(map(torch.randperm, [2] * len_tp_test)))
tp_test = torch.as_tensor(tp_test)[batch_test_idxs, tp_test_perms, :]

torch.save(tp_train, "/home/tol/data/tp_train_"+ test_cell_line)
torch.save(tp_train_perms, "/home/tol/data/tp_train_perms_"+ test_cell_line)
torch.save(tp_valid, "/home/tol/data/tp_valid_"+ test_cell_line)
torch.save(tp_valid_perms, "/home/tol/data/tp_valid_perms_"+ test_cell_line)
torch.save(tp_test, "/home/tol/data/tp_test_"+ test_cell_line)
torch.save(tp_test_perms, "/home/tol/data/tp_test_perms_"+ test_cell_line)
