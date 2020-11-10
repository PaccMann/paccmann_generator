import argparse
import json
import logging
import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from paccmann_chemistry.models import (
    StackGRUDecoder, StackGRUEncoder, TeacherVAE
)
from paccmann_chemistry.utils import get_device
from paccmann_generator.plot_utils import plot_and_compare, plot_and_compare_proteins, plot_loss
from paccmann_generator.utils import add_avg_profile, omics_data_splitter, protein_data_splitter
from paccmann_omics.encoders import ENCODER_FACTORY
from pytoda.smiles.smiles_language import SMILESLanguage
from pytoda.proteins.protein_language import ProteinLanguage
from paccmann_generator.utils import disable_rdkit_logging
from paccmann_predictor.models import MODEL_FACTORY as MODEL_FACTORY_OMICS
import sys
#sys.path.append('/dataP/tol/paccmann_affinity')
sys.path.append('/home/tol/paccmann_affinity')
from paccmann_affinity.models.predictors import MODEL_FACTORY as MODEL_FACTORY_PROTEIN

from paccmann_generator.reinforce_proteins_omics import ReinforceProteinOmics
from paccmann_generator import ReinforceOmic
from paccmann_generator.reinforce_proteins import ReinforceProtein
from files import *
cancer_cell_lines = ['HUH-6-clone5','HuH-7','SNU-475','SNU-423','SNU-387','SNU-449','HLE','C3A']

# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('train_paccmann_rl')
logger_m = logging.getLogger('matplotlib')
logger_m.setLevel(logging.WARNING)

disable_rdkit_logging()

params = dict()
params['site'] = site
params['cancertype'] = cancertype


logger.info(f'Model with name {model_name} starts.')

# Load omics profiles for conditional generation,
# complement with avg per site
omics_df = pd.read_pickle(omics_data_path)
omics_df = add_avg_profile(omics_df)
idx = [i in cancer_cell_lines for i in omics_df['cell_line']]
omics_df  = omics_df[idx]
print("omics data:", omics_df.shape, omics_df['cell_line'].iloc[0])
test_cell_line = omics_df['cell_line'].iloc[0]
#omics_df = omics_df[omics_df.histology == cancertype]

# Load protein sequence data
#if protein_data_path.endswith('.smi'):
#    protein_df = read_smi(protein_data_path, names=['Sequence'])
#elif protein_data_path.endswith('.csv'):
#    protein_df = pd.read_csv(protein_data_path, index_col=0, header=None, names=[str(x) for x in range(768)]) #'entry_name')
#else:
#    raise TypeError(
#        f"{protein_data_path.split('.')[-1]} files are not supported."
#    )

protein_df = pd.read_csv(protein_data_path, index_col=0)#, header=None, names=[str(x) for x in range(768)]) #'entry_name')
protein_df = protein_df[~protein_df.index.isnull()]
protein_df.index = [i.split('|')[2] for i in protein_df.index]
protein_seq_df = pd.read_csv(protein_data_seq_path, names = ['sequence'], index_col=0) #, index_col='entry_name')
#print(protein_seq_df.head)
protein_seq_df.index = [i.split('|')[2] for i in protein_seq_df.index]
protein_df = pd.concat([protein_df, protein_seq_df], axis=1, join='outer')
protein_df = protein_df[[s.split('_')[0] in cancer_genes for s in protein_df.index]]
#print(protein_df.head)
print("proteins:", protein_df.index, len(cancer_genes))

with open(params_path) as f:
    params.update(json.load(f))

# Define network
with open(os.path.join(omics_model_path, 'model_params.json')) as f:
    cell_params = json.load(f)

cell_encoder = ENCODER_FACTORY['dense'](cell_params)
cell_encoder.load(
    os.path.join(
        omics_model_path,
        f"weights/best_{params.get('omics_metric','both')}_encoder.pt"
    ),
    map_location=get_device()
)
cell_encoder.eval()


# Restore SMILES Model
with open(os.path.join(mol_model_path, 'model_params.json')) as f:
    mol_params = json.load(f)

gru_encoder = StackGRUEncoder(mol_params)
gru_decoder = StackGRUDecoder(mol_params)
generator = TeacherVAE(gru_encoder, gru_decoder)
generator.load(
    os.path.join(
        mol_model_path,
        f"weights/best_{params.get('smiles_metric', 'rec')}.pt"
    ),
    map_location=get_device()
)
# Load languages
generator_smiles_language = SMILESLanguage.load(
    os.path.join(mol_model_path, 'selfies_language.pkl')
)
#generator._associate_language(generator_smiles_language)

 # Restore protein model
with open(os.path.join(protein_model_path, 'model_params.json')) as f:
    protein_params = json.load(f)

# Define network
protein_encoder = ENCODER_FACTORY['dense'](protein_params)
protein_encoder.load(
    os.path.join(
        protein_model_path,
        f"weights/best_{params.get('omics_metric','both')}_encoder.pt"
    ),
    map_location=get_device()
)
protein_encoder.eval()

# Restore omics model
with open(os.path.join(omics_model_path, 'model_params.json')) as f:
    cell_params = json.load(f)

# Define network
cell_encoder = ENCODER_FACTORY['dense'](cell_params)
cell_encoder.load(
    os.path.join(
        omics_model_path,
        f"weights/best_{params.get('omics_metric','both')}_encoder.pt"
    ),
    map_location=get_device()
)
cell_encoder.eval()

#load predictors
with open(os.path.join(ic50_model_path, 'model_params.json')) as f:
    paccmann_params = json.load(f)


paccmann_predictor = MODEL_FACTORY_OMICS['mca'](paccmann_params)
paccmann_predictor.load(
    os.path.join(
        ic50_model_path,
        f"weights/best_{params.get('ic50_metric', 'mse')}_mca.pt"
    ),
    map_location=get_device()
)
paccmann_predictor.eval()
paccmann_smiles_language = SMILESLanguage.load(
    os.path.join(ic50_model_path, 'smiles_language.pkl')
)
paccmann_predictor._associate_language(paccmann_smiles_language)

with open(os.path.join(affinity_model_path, 'model_params.json')) as f:
    protein_pred_params = json.load(f)

protein_predictor = MODEL_FACTORY_PROTEIN['bimodal_mca'](protein_pred_params)
protein_predictor.load(
    os.path.join(
        affinity_model_path,
        f"weights/best_{params.get('p_metric', 'ROC-AUC')}_bimodal_mca.pt"
    ),
    map_location=get_device()
)
protein_predictor.eval()

affinity_smiles_language = SMILESLanguage.load(
    os.path.join(affinity_model_path, 'smiles_language.pkl')
)
affinity_protein_language = ProteinLanguage.load(
    os.path.join(affinity_model_path, 'protein_language.pkl')
)
protein_predictor._associate_language(affinity_smiles_language)
protein_predictor._associate_language(affinity_protein_language)

# Specifies the baseline model used for comparison
unbiased_preds_df = pd.read_csv(unbiased_predictions_path)
# yapf: disable

remove_invalid = remove_invalid

# Specifies the baseline model used for comparison
baseline = ReinforceProteinOmics(generator, protein_encoder, cell_encoder, \
    protein_predictor, paccmann_predictor, protein_df, omics_df, \
    params, generator_smiles_language, 'baseline', logger, remove_invalid
)

#############################################
# Create a fresh model that will be optimized
gru_encoder_rl = StackGRUEncoder(mol_params)
gru_decoder_rl = StackGRUDecoder(mol_params)
generator_rl = TeacherVAE(gru_encoder_rl, gru_decoder_rl)
generator_rl.load(
    os.path.join(
        mol_model_path, f"weights/best_{params.get('metric', 'rec')}.pt"
    ),
    map_location=get_device()
)
generator_rl.eval()
#generator_rl._associate_language(generator_smiles_language)

cell_encoder_rl = ENCODER_FACTORY['dense'](cell_params)
cell_encoder_rl.load(
    os.path.join(
        omics_model_path,
        f"weights/best_{params.get('metric', 'both')}_encoder.pt"
    ),
    map_location=get_device()
)
cell_encoder_rl.eval()

protein_encoder_rl = ENCODER_FACTORY['dense'](protein_params)
protein_encoder_rl.load(
    os.path.join(
        protein_model_path,
        f"weights/best_{params.get('metric', 'both')}_encoder.pt"
    ),
    map_location=get_device()
)
protein_encoder_rl.eval()

model_folder_name = site + '_' + model_name + '_combined'
learner_combined = ReinforceProteinOmics(
    generator_rl, protein_encoder_rl,cell_encoder_rl, \
    protein_predictor, paccmann_predictor, protein_df, omics_df, \
    params, generator_smiles_language, model_folder_name, logger, remove_invalid
)

gru_encoder_rl_o = StackGRUEncoder(mol_params)
gru_decoder_rl_o = StackGRUDecoder(mol_params)
generator_rl_o = TeacherVAE(gru_encoder_rl_o, gru_decoder_rl_o)
generator_rl_o.load(
    os.path.join(
        mol_model_path, f"weights/best_{params.get('metric', 'rec')}.pt"
    ),
    map_location=get_device()
)
generator_rl_o.eval()
#generator_rl._associate_language(generator_smiles_language)

cell_encoder_rl_o = ENCODER_FACTORY['dense'](cell_params)
cell_encoder_rl_o.load(
    os.path.join(
        omics_model_path,
        f"weights/best_{params.get('metric', 'both')}_encoder.pt"
    ),
    map_location=get_device()
)
cell_encoder_rl_o.eval()

model_folder_name = site + '_' + model_name + '_omics'
learner_omics = ReinforceOmic(
    generator_rl_o, cell_encoder_rl_o, paccmann_predictor, omics_df, params,
    generator_smiles_language, model_folder_name, logger, remove_invalid
)

gru_encoder_rl_p = StackGRUEncoder(mol_params)
gru_decoder_rl_p = StackGRUDecoder(mol_params)
generator_rl_p = TeacherVAE(gru_encoder_rl_p, gru_decoder_rl_p)
generator_rl_p.load(
    os.path.join(
        mol_model_path, f"weights/best_{params.get('metric', 'rec')}.pt"
    ),
    map_location=get_device()
)
generator_rl_p.eval()
#generator_rl._associate_language(generator_smiles_language)

protein_encoder_rl_p = ENCODER_FACTORY['dense'](protein_params)
protein_encoder_rl_p.load(
    os.path.join(
        protein_model_path,
        f"weights/best_{params.get('metric', 'both')}_encoder.pt"
    ),
    map_location=get_device()
)
protein_encoder_rl_p.eval()

model_folder_name = site + '_' + model_name + '_protein'
learner_protein = ReinforceProtein(
    generator_rl_p, protein_encoder_rl_p, protein_predictor, protein_df, params,
    generator_smiles_language, model_folder_name, logger, remove_invalid
)

# Split the samples for conditional generation and initialize training
train_omics, test_omics = omics_data_splitter(
    omics_df, site, params.get('test_fraction', 0.2)
)
train_protein, test_protein = protein_data_splitter(
    protein_df, params.get('test_fraction', 0.2)
)
print("test omics:", len(test_omics), "test protein", test_protein.shape, "train omics:", len(train_omics), "train protein:", train_protein.shape)

rewards, rl_losses, rewards_p, losses_p, rewards_o, losses_o = [], [], [], [], [], []
gen_mols ,gen_prot, gen_affinity, gen_cell, gen_ic50, modes = [], [], [], [], [], []
gen_mols_o ,gen_cell_o, gen_ic50_o, modes_o = [], [], [], []
gen_mols_p ,gen_prot_p, gen_affinity_p, modes_p = [], [], [], []

logger.info('Models restored, start training.')

# choose a validation cell line and protein.
#eval_cell_lines = np.random.choice(test_omics, size = 20, replace=True)
#eval_cell_lines = [str(i) for i in eval_cell_lines]
eval_cell_line = test_cell_line
eval_protein_names = np.random.choice(test_protein.index, size = 20, replace=False)
eval_protein_names = [str(i) for i in eval_protein_names]

for epoch in range(1, params['epochs'] + 1):

    for step in range(1, params['steps'] + 1):

        # Randomly sample a cell lines and proteins for training:
        #cell_line = np.random.choice(train_omics, size = 20, replace=True)
        #cell_line = [str(i) for i in cell_line]
        cell_line  = [i for i in omics_df['cell_line'] if i != test_cell_line]
        protein_name = np.random.choice(train_protein.index, size = 20, replace=False)
        protein_name = [str(i) for i in protein_name]
        #cell_line = [np.random.choice(train_omics)]
        print(f'Current train cell_lines: {cell_line}')
        # Randomly sample a protein
        #protein_name = [np.random.choice(train_protein.index)]
        print(f'Current train proteins: {protein_name}')

        rew, loss = learner_combined.policy_gradient(
            protein_name, cell_line, epoch, params['batch_size']
        )
        rew_p, loss_p = learner_protein.policy_gradient(
            protein_name, epoch, params['batch_size']
        )
        rew_o, loss_o = learner_omics.policy_gradient(
            cell_line, epoch, params['batch_size']
        )
        print(
            f"Epoch {epoch:d}/{params['epochs']:d}, step {step:d}/"
            f"{params['steps']:d}\t loss={loss:.2f}, rew={rew:.2f}"
        )

        rewards.append(rew.item())
        rl_losses.append(loss)
        rewards_p.append(rew_p.item())
        losses_p.append(loss_p)
        rewards_o.append(rew_o.item())
        losses_o.append(loss_o)

    # Save model
    learner_combined.save(f'gen_{epoch}.pt', f'enc_{epoch}_protein.pt', f'enc_{epoch}_omics.pt')
    learner_omics.save(f'gen_{epoch}.pt', f'enc_{epoch}.pt')
    learner_protein.save(f'gen_{epoch}.pt', f'enc_{epoch}.pt')
    unbiased_predsP = unbiased_preds_df[protein_name].values.reshape(-1)[:params['batch_size']]
    unbiased_predsO = unbiased_preds_df[cell_line].values.reshape(-1)[:params['batch_size']]
    base_smiles, base_predsP, base_predsO, idx = baseline.generate_compounds_and_evaluate(
        epoch, params['batch_size'], protein_name, cell_line
    )
    smiles_o, preds_o, idx_o = learner_omics.generate_compounds_and_evaluate(
        epoch, params['batch_size'], cell_line
    )
    smiles, predsP, predsO, idx_c = learner_combined.generate_compounds_and_evaluate(
        epoch, params['batch_size'], protein_name, cell_line
    )
    smiles_p, preds_p, idx_p = (
        learner_protein.generate_compounds_and_evaluate(
            epoch, params['batch_size'], protein_name
        )
    )
    proteins = [val for i, val in enumerate(protein_name*params["batch_size"]) if i in idx_c]
    cell = [val for i, val in enumerate(cell_line*params["batch_size"]) if i in idx_c]
    gs = [
        s for i, s in enumerate(smiles)
        if predsO[i] < learner_combined.ic50_threshold and predsP[i] > 0.5
    ]
    gp_o = predsO[(predsO < learner_combined.ic50_threshold) & (predsP > 0.5)]
    gp_p = predsP[(predsO < learner_combined.ic50_threshold) & (predsP > 0.5)]
    for p_o, p_p, s, p, c in zip(gp_o, gp_p, gs, proteins, cell):
        gen_mols.append(s)
        gen_cell.append(c)
        gen_prot.append(p)
        gen_affinity.append(p_p)
        gen_ic50.append(p_o)
        modes.append('train')

    inds = np.argsort(gp_o)[::-1]
    for i in inds[:5]:
        logger.info(
            f'Epoch {epoch:d}, generated {gs[i]} against '
            f'protein {gen_prot[i]} and cell line {gen_cell[i]}.\n Predicted IC50 = {gp_o[i]} and Affinity = {gp_p[i]}. '
        )

    plot_and_compare(
        unbiased_predsO, predsO, site, cell_line, epoch, learner_combined.model_path,
        'train_combined', params['batch_size']
    )
    plot_and_compare_proteins(
        unbiased_predsP, predsP, protein_name, epoch, learner_combined.model_path,
        'train_combined', params['batch_size']
    )
    #print(len(smiles), len(preds_o), len(predsO), len(preds_p))
    cell = [val for i, val in enumerate(cell_line*params["batch_size"]) if i in idx_o]
    gs_omics = [
        s for i, s in enumerate(smiles_o)
        if preds_o[i] < learner_omics.ic50_threshold
    ]
    gp_omics = preds_o[preds_o < learner_omics.ic50_threshold]
    for p, s, c in zip(gp_omics, gs_omics, cell):
        gen_mols_o.append(s)
        gen_cell_o.append(c)
        gen_ic50_o.append(p)
        modes_o.append('train')

    plot_and_compare(
        unbiased_predsO, preds_o, site, cell_line, epoch, learner_omics.model_path,
        'train_omics', params['batch_size']
    )

    proteins = [val for i, val in enumerate(protein_name*params["batch_size"]) if i in idx_p]
    gs_protein = [s for i, s in enumerate(smiles_p) if preds_p[i] > 0.5]
    gp_protein = preds_p[preds_p > 0.5]
    for p, s, prot in zip(gp_protein, gs_protein, proteins):
        gen_mols_p.append(s)
        gen_prot_p.append(prot)
        gen_affinity_p.append(p)
        modes_p.append('train')

    plot_and_compare_proteins(
            unbiased_predsP, preds_p, protein_name, epoch, learner_protein.model_path,
            'train_protein', params['batch_size']
        )
    
    # Evaluate on a validation cell line and protein.
    unbiased_predsP = unbiased_preds_df[eval_protein_names].values.reshape(-1)[:params['eval_batch_size']]
    unbiased_predsO = unbiased_preds_df[eval_cell_lines].values.reshape(-1)[:params['eval_batch_size']]
    base_smiles, base_predsP, base_predsO, idx = baseline.generate_compounds_and_evaluate(
        epoch, params['eval_batch_size'], eval_protein_names, eval_cell_lines
    )
    smiles_o, preds_o, idx_o = learner_omics.generate_compounds_and_evaluate(
        epoch, params['eval_batch_size'], eval_cell_lines
    )
    smiles, predsP, predsO, idx_c = learner_combined.generate_compounds_and_evaluate(
        epoch, params['eval_batch_size'], eval_protein_names, eval_cell_lines
    )
    smiles_p, preds_p, idx_p = (
        learner_protein.generate_compounds_and_evaluate(
            epoch, params['eval_batch_size'], eval_protein_names
        )
    )
    
    plot_and_compare(
        unbiased_predsO, predsO, site, eval_cell_lines, epoch, learner_combined.model_path,
        'test_combined', params['eval_batch_size']
    )
    
    plot_and_compare_proteins(
        unbiased_predsP, predsP, eval_protein_names, epoch, learner_combined.model_path,
        'test_combined', params['eval_batch_size']
    )

    plot_and_compare(
        unbiased_predsO, preds_o, site, eval_cell_lines, epoch, learner_omics.model_path,
        'test_omics', params['eval_batch_size']
    )
    
    plot_and_compare_proteins(
        unbiased_predsP, preds_p, eval_protein_names, epoch, learner_protein.model_path,
        'test_protein', params['eval_batch_size']
    )
    proteins = [val for i, val in enumerate(eval_protein_names*params["eval_batch_size"]) if i in idx_c]
    cell = [val for i, val in enumerate(eval_cell_lines*params["eval_batch_size"]) if i in idx_c]

    gs = [
        s for i, s in enumerate(smiles)
        if predsO[i] < learner_combined.ic50_threshold and predsP[i] > 0.5
    ]
    gp_o = predsO[(predsO < learner_combined.ic50_threshold) & (predsP > 0.5)]
    gp_p = predsP[(predsO < learner_combined.ic50_threshold) & (predsP > 0.5)]

    for p_o, p_p, s, c, p in zip(gp_o, gp_p, gs, cell, proteins):
        gen_mols.append(s)
        gen_cell.append(c)
        gen_prot.append(p)
        gen_affinity.append(p_p)
        gen_ic50.append(p_o)
        modes.append('test')

    cell = [val for i, val in enumerate(eval_cell_lines*params["eval_batch_size"]) if i in idx_o]
    gs_omics = [
        s for i, s in enumerate(smiles_o)
        if preds_o[i] < learner_omics.ic50_threshold
    ]
    gp_omics = preds_o[preds_o < learner_omics.ic50_threshold]
    for p, s, c in zip(gp_omics, gs_omics, cell):
        gen_mols_o.append(s)
        gen_cell_o.append(c)
        gen_ic50_o.append(p)
        modes_o.append('test')

    proteins = [val for i, val in enumerate(eval_protein_names*params["eval_batch_size"]) if i in idx_p]
    gs_protein = [s for i, s in enumerate(smiles_p) if preds_p[i] > 0.5]
    gp_protein = preds_p[preds_p > 0.5]
    for p, s, prot in zip(gp_protein, gs_protein, proteins):
        gen_mols_p.append(s)
        gen_prot_p.append(prot)
        gen_affinity_p.append(p)
        modes_p.append('test')
    
    inds = np.argsort(gp_o)[::-1]
    for i in inds[:5]:
        logger.info(
            f'Epoch {epoch:d}, generated {gs[i]} against '
            f'{gen_cell[i]} and protein {gen_prot[i]}.\n Predicted IC50 = {gp_o[i]}and Affinity = {gp_p[i]}. '
        )

    # Save results (good molecules!) in DF
    df = pd.DataFrame(
        {
            'protein': gen_prot,
            'cell_line': gen_cell,
            'SMILES': gen_mols,
            'IC50': gen_ic50,
            'Binding probability': gen_affinity,
            'mode': modes,
            'tox21': [learner_combined.tox21(s) for s in gen_mols],
            'epoch': [epoch]*len(gen_mols), 
            'validity': [round((len(predsO)/params['batch_size']) * 100, 1)]*len(gen_mols)
        }
    )
    if epoch ==1:
        df.to_csv(os.path.join(learner_combined.model_path, 'results', 'generated.csv'))
    else:
        df.to_csv(os.path.join(learner_combined.model_path, 'results', 'generated.csv'), mode='a', header=False)
    # Plot loss development
    loss_df = pd.DataFrame({'loss': rl_losses, 'rewards': rewards, 'epoch':epoch})
    if epoch ==1:
        loss_df.to_csv(learner_combined.model_path + '/results/loss_reward_evolution.csv')
    else:
        loss_df.to_csv(learner_combined.model_path + '/results/loss_reward_evolution.csv', mode='a', header=False)
    
    plot_loss(
        rl_losses,
        rewards,
        params['epochs'],
        #cell_line + ',' + protein_name,
        learner_combined.model_path,
        rolling=5,
        site=site
    )

    df = pd.DataFrame(
        {
            'cell_line': gen_cell_o,
            'SMILES': gen_mols_o,
            'IC50': gen_ic50_o,
            'mode': modes_o,
            'tox21': [learner_omics.tox21(s) for s in gen_mols_o],
            'epoch': [epoch]*len(gen_mols_o), 
            'validity': [round((len(preds_o)/params['batch_size']) * 100, 1)]*len(gen_mols_o)
        }
    )
    if epoch ==1:
        df.to_csv(os.path.join(learner_omics.model_path, 'results', 'generated.csv'))
    else:
        df.to_csv(os.path.join(learner_omics.model_path, 'results', 'generated.csv'), mode='a', header=False)
    # Plot loss development
    loss_df = pd.DataFrame({'loss': losses_o, 'rewards': rewards_o, 'epoch':epoch})
    if epoch ==1:
        loss_df.to_csv(learner_omics.model_path + '/results/loss_reward_evolution.csv')
    else:
        loss_df.to_csv(learner_omics.model_path + '/results/loss_reward_evolution.csv', mode='a', header=False)
    plot_loss(
        losses_o,
        rewards_o,
        params['epochs'],
        #cell_line,
        learner_omics.model_path,
        rolling=5,
        site=site
    )

    df = pd.DataFrame(
        {
            'protein': gen_prot_p,
            'SMILES': gen_mols_p,
            'Binding probability': gen_affinity_p,
            'mode': modes_p,
            'epoch': [epoch]*len(gen_mols_p), 
            'validity': [round((len(preds_p)/params['batch_size']) * 100, 1)]*len(gen_mols_p)
        }
    )
    if epoch ==1:
        df.to_csv(os.path.join(learner_protein.model_path, 'results', 'generated.csv'))
    else:
        df.to_csv(os.path.join(learner_protein.model_path, 'results', 'generated.csv'), mode='a', header=False)
    # Plot loss development
    loss_df = pd.DataFrame({'loss': losses_p, 'rewards': rewards_p, 'epoch':epoch})
    if epoch ==1:
        loss_df.to_csv(learner_protein.model_path + '/results/loss_reward_evolution.csv')
    else:
        loss_df.to_csv(learner_protein.model_path + '/results/loss_reward_evolution.csv', mode='a', header=False)
    plot_loss(
        losses_p,
        rewards_p,
        params['epochs'],
        #protein_name,
        learner_protein.model_path,
        rolling=5
    )