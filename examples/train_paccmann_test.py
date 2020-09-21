import argparse
import json
import logging
import os
import sys
import numpy as np
import pandas as pd
from paccmann_chemistry.models import (
    StackGRUDecoder, StackGRUEncoder, TeacherVAE
)
from paccmann_chemistry.utils import get_device
from paccmann_generator.plot_utils import plot_and_compare, plot_and_compare_proteins, plot_loss
from paccmann_generator.utils import add_avg_profile, omics_data_splitter
from paccmann_omics.encoders import ENCODER_FACTORY
from pytoda.smiles.smiles_language import SMILESLanguage
from pytoda.proteins.protein_language import ProteinLanguage
from paccmann_generator.utils import disable_rdkit_logging
from paccmann_predictor.models import MODEL_FACTORY as MODEL_FACTORY_OMICS
import sys
sys.path.append('/home/tol/paccmann_affinity')
from paccmann_affinity.models.predictors import MODEL_FACTORY as MODEL_FACTORY_PROTEIN

from paccmann_generator.reinforce_proteins_omics import ReinforceProteinOmics

# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('train_paccmann_rl')
logger_m = logging.getLogger('matplotlib')
logger_m.setLevel(logging.WARNING)

language_path = '/mnt/c/Users/PatriciaStoll/paccmann_affinity/trained_models/vae_selfies_one_hot_mod'
mol_model_path = '/mnt/c/Users/PatriciaStoll/paccmann_affinity/trained_models/vae_selfies_one_hot'
omics_model_path = 'paccmann_generator/pvae'
protein_model_path = '/mnt/c/Users/PatriciaStoll/paccmann_affinity/trained_models/pevae_avg'
ic50_model_path = 'paccmann'
affinity_model_path = '/mnt/c/Users/PatriciaStoll/paccmann_affinity/trained_models/base_affinity'
omics_data_path = 'data/gdsc_transcriptomics_for_conditional_generation.pkl'
protein_data_path = '/mnt/c/Users/PatriciaStoll/paccmann_affinity/sars_cov2_data/tape/transformer/avg.csv'
protein_data_seq_path = '/mnt/c/Users/PatriciaStoll/paccmann_affinity/sars_cov2_data/uniprot_sars_cov2.csv'
params_path = 'examples/example_params.json'
model_name = 'test_C_frac_higher_weight'
site = 'lung'


disable_rdkit_logging()

params = dict()
params['site'] = site


logger.info(f'Model with name {model_name} starts.')

# Load omics profiles for conditional generation,
# complement with avg per site
omics_df = pd.read_pickle(omics_data_path)
omics_df = add_avg_profile(omics_df)

# Load protein sequence data
#if protein_data_path.endswith('.smi'):
#    protein_df = read_smi(protein_data_path, names=['Sequence'])
#elif protein_data_path.endswith('.csv'):
#    protein_df = pd.read_csv(protein_data_path, index_col=0, header=None, names=[str(x) for x in range(768)]) #'entry_name')
#else:
#    raise TypeError(
#        f"{protein_data_path.split('.')[-1]} files are not supported."
#    )

protein_df = pd.read_csv(protein_data_path, index_col=0, header=None, names=[str(x) for x in range(768)]) #'entry_name')
protein_df.index = [i.split('|')[2] for i in protein_df.index]
protein_seq_df = pd.read_csv(protein_data_seq_path, index_col='entry_name')
protein_df = pd.concat([protein_df, protein_seq_df], axis=1, join='outer')

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
        f"weights/best_{params.get('ic50_metric', 'rmse')}_mca.pt"
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
baseline = ReinforceProteinOmics(generator, protein_encoder, cell_encoder, \
    protein_predictor, paccmann_predictor, protein_df, omics_df, \
    params, generator_smiles_language, 'baseline', logger
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

model_folder_name = site + '_' + model_name
learner = ReinforceProteinOmics(
    generator_rl, protein_encoder_rl,cell_encoder_rl, \
    protein_predictor, paccmann_predictor, protein_df, omics_df, \
    params, generator_smiles_language, model_folder_name, logger
)



# Split the samples for conditional generation and initialize training
train_omics, test_omics = omics_data_splitter(
    omics_df, site, params.get('test_fraction', 0.2)
)
rewards, rl_losses = [], []
gen_mols ,gen_prot, gen_affinity, gen_cell, gen_ic50, modes = [], [], [], [], [], []

logger.info('Models restored, start training.')

for epoch in range(1, params['epochs'] + 1):

    for step in range(1, params['steps'] + 1):

        # Randomly sample a cell line:
        cell_line = np.random.choice(train_omics)
        print(f'Current train cell_line: {cell_line}')
        # Randomly sample a protein
        protein_name = np.random.choice(protein_df.index)
        print(f'Current train protein: {protein_name}')

        rew, loss = learner.policy_gradient(
            protein_name, cell_line, epoch, params['batch_size']
        )
        print(
            f"Epoch {epoch:d}/{params['epochs']:d}, step {step:d}/"
            f"{params['steps']:d}\t loss={loss:.2f}, rew={rew:.2f}"
        )

        rewards.append(rew.item())
        rl_losses.append(loss)

    # Save model
    learner.save(f'gen_{epoch}.pt', f'enc_{epoch}.pt')
    print(f'EVAL protein: {protein_name}')

    # Compare baseline and trained model on cell line
    base_smiles, base_predsP, base_predsO = baseline.generate_compounds_and_evaluate(
        epoch, params['eval_batch_size'], protein_name, cell_line
    )
    smiles, predsP, predsO = learner.generate_compounds_and_evaluate(
        epoch, params['eval_batch_size'], protein_name, cell_line
    )
    gs = [
        s for i, s in enumerate(smiles)
        if predsO[i] < learner.ic50_threshold and predsP[i] > 0.5
    ]
    gp_o = predsO[(predsO < learner.ic50_threshold) & (predsP > 0.5)]
    gp_p = predsP[(predsO < learner.ic50_threshold) & (predsP > 0.5)]

    for p_o, p_p, s in zip(gp_o, gp_p, gs):
        gen_mols.append(s)
        gen_cell.append(cell_line)
        gen_prot.append(protein_name)
        gen_affinity.append(p_p)
        gen_ic50.append(p_o)
        modes.append('train')

    inds = np.argsort(gp_o)[::-1]
    for i in inds[:5]:
        logger.info(
            f'Epoch {epoch:d}, generated {gs[i]} against '
            f'protein {protein_name} and cell line {cell_line}.\n Predicted IC50 = {gp_o[i]} and Affinity = {gp_p[i]}. '
        )

    plot_and_compare(
        base_predsO, predsO, site, cell_line, epoch, learner.model_path,
        'train', params['eval_batch_size']
    )
    
    plot_and_compare_proteins(
        base_predsP, predsP, protein_name, epoch, learner.model_path,
        'train', params['eval_batch_size']
    )

    
    # Evaluate on a validation cell line.
    eval_cell_line = np.random.choice(test_omics)
    base_smiles, base_predsP, base_predsO = baseline.generate_compounds_and_evaluate(
        epoch, params['eval_batch_size'], protein_name, eval_cell_line
    )
    smiles, predsP, predsO = learner.generate_compounds_and_evaluate(
        epoch, params['eval_batch_size'], protein_name, eval_cell_line
    )
    
    plot_and_compare(
        base_predsO, predsO, site, cell_line, epoch, learner.model_path,
        'test', params['eval_batch_size']
    )

    gs = [
        s for i, s in enumerate(smiles)
        if predsO[i] < learner.ic50_threshold and predsP[i] > 0.5
    ]
    gp_o = predsO[(predsO < learner.ic50_threshold) & (predsP > 0.5)]
    gp_p = predsP[(predsO < learner.ic50_threshold) & (predsP > 0.5)]

    for p_o, p_p, s in zip(gp_o, gp_p, gs):
        gen_mols.append(s)
        gen_cell.append(cell_line)
        gen_prot.append(protein_name)
        gen_affinity.append(p_p)
        gen_ic50.append(p_o)
        modes.append('test')

    inds = np.argsort(gp_o)[::-1]
    for i in inds[:5]:
        logger.info(
            f'Epoch {epoch:d}, generated {gs[i]} against '
            f'{eval_cell_line} and protein {protein_name}.\n Predicted IC50 = {gp_o[i]}and Affinity = {gp_p[i]}. '
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
            'tox21': [learner.tox21(s) for s in gen_mols]
        }
    )
    df.to_csv(os.path.join(learner.model_path, 'results', 'generated.csv'))
    # Plot loss development
    loss_df = pd.DataFrame({'loss': rl_losses, 'rewards': rewards})
    loss_df.to_csv(
        learner.model_path + '/results/loss_reward_evolution.csv'
    )
    plot_loss(
        rl_losses,
        rewards,
        params['epochs'],
        cell_line + ',' + protein_name,
        learner.model_path,
        rolling=5,
        site=site
    )