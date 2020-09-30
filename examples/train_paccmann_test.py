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
from paccmann_generator.utils import add_avg_profile, omics_data_splitter, protein_data_splitter
from paccmann_omics.encoders import ENCODER_FACTORY
from pytoda.smiles.smiles_language import SMILESLanguage
from pytoda.proteins.protein_language import ProteinLanguage
from paccmann_generator.utils import disable_rdkit_logging
from paccmann_predictor.models import MODEL_FACTORY as MODEL_FACTORY_OMICS
import sys
sys.path.append('/home/tol/paccmann_affinity')
from paccmann_affinity.models.predictors import MODEL_FACTORY as MODEL_FACTORY_PROTEIN

from paccmann_generator.reinforce_proteins_omics import ReinforceProteinOmics
from paccmann_generator import ReinforceOmic
from paccmann_generator.reinforce_proteins import REINFORCE_proteins

# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('train_paccmann_rl')
logger_m = logging.getLogger('matplotlib')
logger_m.setLevel(logging.WARNING)

language_path = '/mnt/c/Users/PatriciaStoll/paccmann_affinity/trained_models/vae_selfies_one_hot_mod'
mol_model_path = '/mnt/c/Users/PatriciaStoll/paccmann_affinity/trained_models/vae_selfies_one_hot'
omics_model_path = 'pvae'
protein_model_path = '/mnt/c/Users/PatriciaStoll/paccmann_affinity/trained_models/pevae_avg'
ic50_model_path = '../paccmann_003'
affinity_model_path = '/mnt/c/Users/PatriciaStoll/paccmann_affinity/trained_models/base_affinity'
omics_data_path = 'data/gdsc_transcriptomics_for_conditional_generation.pkl'
protein_data_path = '/mnt/c/Users/PatriciaStoll/Documents/data/embedding.csv'#paccmann_affinity/sars_cov2_data/tape/transformer/avg.csv'
protein_data_seq_path = '/mnt/c/Users/PatriciaStoll/Documents/data/uniprot-yourlist M20200922A94466D2655679D1FD8953E075198DA87EBCA61-filtered-rev--.fasta'
params_path = 'examples/example_params.json'
model_name = 'multitest'
site = 'lung'
cancertype = 'neuroblastoma'
cancer_genes = ['RUNX1','CSF1R','MPO','CSF2','IL3','RUNX1T1','HDAC1','HDAC2','SIN3A','NCOR1','CEBPA','PER2','SPI1','CD14','ITGAM','FCGR1A','JUP','PML','RARA','CCNA2','CCNA1','CEBPE','BCL2A1','ZBTB16','MYC','DUSP6','TCF3','PBX1','WNT16','ETV6','ETV7','DEFA1','DEFA3','DEFA4','DEFA5','DEFA6','DEFA1B','ELANE','GZMB','KMT2A','AFF1','CDK9','CCNT1','CCNT2','MLLT1','MLLT3','DOT1L','LMO2','PBX3','RUNX2','SMAD1','KLF3','MEF2C','HOXA9','HOXA10','JMJD1C','HMGA2','KDM6A','SUPT3H','PROM1','FLT3','BMP2K','IGF1R','CDKN1B','CDK14','MEIS1','HOXA11','SIX1','SIX4','EYA1','CDKN2C','HPGD','GRIA3','FUT8','TLX3','TLX1','BCL11B','LDB1','LYL1','HHEX','PTCRA','REL','CCND2','BIRC2','BIRC3','TRAF1','BCL2L1','CD86','CD40','BCL6','IGH','MAF','ITGB7','NSD2','H3-5','H3-3B','H3C4','H3C3','H3C1','H3-3A','H3-4','H3C14','H3C15','H3C13','H3C6','H3C11','H3C8','H3C12','H3C10','H3C2','H3C7','PAX5','PAX8','PPARG','RXRA','RXRB','RXRG','PRCC','TFE3','CDKN1A','TMPRSS2','ERG','PLAU','PLAT','MMP3','MMP9','ZEB1','IL1R2','SPINT1','ETV1','ETV4','ETV5','SLC45A3','ELK4','DDX5','MYCN','MAX','MDM2','PTK2','TP53','BMI1','COMMD3-BMI1','SP1','ZBTB17','NTRK1','NGFR','MEN1','EWSR1','FLI1','IGF1','ID2','TGFBR2','IGFBP3','FEV','ATF1','ARNT2','ATM','MITF','WT1','PDGFA','IL2RB','BAIAP3','TSPAN7','MLF1','NR4A3','TAF15','FUS','DDIT3','CEBPB','IL6','NFKBIZ','NFKB1','RELA','CXCL8','PAX7','PAX3','FOXO1','FLT1','SS18','SSX1','SSX2','SSX2B','NUPR1','ASPSCR1','MET','GADD45A','GADD45B','GADD45G','BAX','BAK1','DDB2','POLK']

disable_rdkit_logging()

params = dict()
params['site'] = site
params['cancertype'] = cancertype


logger.info(f'Model with name {model_name} starts.')

# Load omics profiles for conditional generation,
# complement with avg per site
omics_df = pd.read_pickle(omics_data_path)
omics_df = add_avg_profile(omics_df)
omics_df = omics_df[omics_df.histology == cancertype]

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
#print(protein_df.shape)
protein_df.index = [i.split('|')[2] for i in protein_df.index]
protein_seq_df = pd.read_csv(protein_data_seq_path, index_col='entry_name')
#protein_df = pd.concat([protein_df, protein_seq_df], axis=1, join='outer')

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
learner_combined = ReinforceProteinOmics(
    generator_rl, protein_encoder_rl,cell_encoder_rl, \
    protein_predictor, paccmann_predictor, protein_df, omics_df, \
    params, generator_smiles_language, model_folder_name, logger
)
learner_protein = REINFORCE_proteins(
    generator_rl, protein_encoder_rl, predictor, protein_df, params,
    model_folder_name, logger
)
learner_omics = ReinforceOmic(
    generator_rl, cell_encoder_rl, paccmann_predictor, omics_df, params,
    model_folder_name, logger
)

# Split the samples for conditional generation and initialize training
train_omics, test_omics = omics_data_splitter(
    omics_df, site, params.get('test_fraction', 0.2)
)
train_protein, test_protein = protein_data_splitter(
    protein_df, params.get('test_fraction', 0.2)
)

rewards, rl_losses, rewards_p, losses_p, rewards_o, losses_o = [], [], [], [], [], []
gen_mols ,gen_prot, gen_affinity, gen_cell, gen_ic50, modes = [], [], [], [], [], []
gen_mols_o ,gen_cell, gen_ic50, modes_o = [], [], [], []
gen_mols_p ,gen_prot_p, gen_affinity_p, modes_p = [], [], [], []

logger.info('Models restored, start training.')

# choose a validation cell line and protein.
eval_cell_lines = np.random.choice(test_omics, size = 20, replace=True)
eval_protein_names = np.random.choice(test_protein.index, size = 20, replace=False)

for epoch in range(1, params['epochs'] + 1):

    for step in range(1, params['steps'] + 1):

        # Randomly sample a cell line:
        cell_line = [np.random.choice(train_omics)]
        print(f'Current train cell_line: {cell_line}')
        # Randomly sample a protein
        protein_name = [np.random.choice(train_protein.index)]
        print(f'Current train protein: {protein_name}')

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
    learner_combined.save(f'gen_{epoch}.pt', f'enc_{epoch}.pt')
    learner_omics.save(f'gen_{epoch}.pt', f'enc_{epoch}.pt')
    learner_protein.save(f'gen_{epoch}.pt', f'enc_{epoch}.pt')

    base_smiles, base_predsP, base_predsO = baseline.generate_compounds_and_evaluate(
        epoch, params['eval_batch_size'], protein_name, cell_line
    )
    smiles_o, preds_o = learner_omics.generate_compounds_and_evaluate(
        epoch, params['eval_batch_size'], cell_line
    )
    smiles, predsP, predsO = learner_combined.generate_compounds_and_evaluate(
        epoch, params['eval_batch_size'], protein_name, cell_line
    )
    smiles_p, preds_p = (
        learner_protein.generate_compounds_and_evaluate(
            epoch, params['eval_batch_size'], protein_name
        )
    )
    gs = [
        s for i, s in enumerate(smiles)
        if predsO[i] < learner_combined.ic50_threshold and predsP[i] > 0.5
    ]
    gp_o = predsO[(predsO < learner_combined.ic50_threshold) & (predsP > 0.5)]
    gp_p = predsP[(predsO < learner_combined.ic50_threshold) & (predsP > 0.5)]

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
        base_predsO, predsO, site, cell_line, epoch, learner_combined.model_path,
        'train_combined', params['eval_batch_size']
    )
    plot_and_compare_proteins(
        base_predsP, predsP, protein_name, epoch, learner_combined.model_path,
        'train_combined', params['eval_batch_size']
    )
    
    gs_omics = [
        s for i, s in enumerate(smiles)
        if preds_o[i] < learner_omics.ic50_threshold
    ]
    gp_omics = preds_o[preds_o < learner_omics.ic50_threshold]
    for p, s in zip(gp_omics, gs_omics):
        gen_mols_o.append(s)
        gen_cell_o.append(cell_line)
        gen_ic50_o.append(p)
        modes_o.append('train')

    plot_and_compare(
        base_predsO, preds_o, site, cell_line, epoch, learner_omics.model_path,
        'train_omics', params['eval_batch_size']
    )

    gs_protein = [s for i, s in enumerate(smiles) if preds_p[i] > 0.5]
    gp_protein = preds_p[preds_p > 0.5]
    for p, s in zip(gp_protein, gs_protein):
        gen_mols_p.append(s)
        gen_prot_p.append(protein_name)
        gen_affinity_p.append(p)
        modes_p.append('train')

    plot_and_compare_proteins(
            base_predsP, preds_p, protein_name, epoch, learner_protein.model_path,
            'train_protein', params['eval_batch_size']
        )
    
    # Evaluate on a validation cell line and protein.
    base_smiles, base_predsP, base_predsO = baseline.generate_compounds_and_evaluate(
        epoch, params['eval_batch_size'], eval_protein_names, eval_cell_lines
    )
    smiles_o, preds_o = learner_omics.generate_compounds_and_evaluate(
        epoch, params['eval_batch_size'], eval_cell_lines
    )
    smiles, predsP, predsO = learner_combined.generate_compounds_and_evaluate(
        epoch, params['eval_batch_size'], eval_protein_names, eval_cell_lines
    )
    smiles_p, preds_p = (
        learner_protein.generate_compounds_and_evaluate(
            epoch, params['eval_batch_size'], eval_protein_names
        )
    )
    
    plot_and_compare(
        base_predsO, predsO, site, eval_cell_lines, epoch, learner_combined.model_path,
        'test_combined', params['eval_batch_size']
    )
    
    plot_and_compare_proteins(
        base_predsP, predsP, eval_protein_names, epoch, learner_combined.model_path,
        'test_combined', params['eval_batch_size']
    )

    plot_and_compare(
        base_predsO, preds_o, site, eval_cell_lines, epoch, learner_omics.model_path,
        'test_omics', params['eval_batch_size']
    )
    
    plot_and_compare_proteins(
        base_predsP, preds_p, eval_protein_names, epoch, learner_protein.model_path,
        'test_protein', params['eval_batch_size']
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

    gs_omics = [
        s for i, s in enumerate(smiles)
        if preds_o[i] < learner_omics.ic50_threshold
    ]
    gp_omics = preds_o[preds_o < learner_omics.ic50_threshold]
    for p, s in zip(gp_omics, gs_omics):
        gen_mols_o.append(s)
        gen_cell_o.append(cell_line)
        gen_ic50_o.append(p)
        modes_o.append('train')

    gs_protein = [s for i, s in enumerate(smiles) if preds_p[i] > 0.5]
    gp_protein = preds_p[preds_p > 0.5]
    for p, s in zip(gp_protein, gs_protein):
        gen_mols_p.append(s)
        gen_prot_p.append(protein_name)
        gen_affinity_p.append(p)
        modes_p.append('train')
    
    inds = np.argsort(gp_o)[::-1]
    for i in inds[:5]:
        logger.info(
            f'Epoch {epoch:d}, generated {gs[i]} against '
            f'{eval_cell_lines} and protein {protein_name}.\n Predicted IC50 = {gp_o[i]}and Affinity = {gp_p[i]}. '
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
            'tox21': [learner_combined.tox21(s) for s in gen_mols]
        }
    )
    df.to_csv(os.path.join(learner_combined.model_path, 'results', 'generated.csv'))
    # Plot loss development
    loss_df = pd.DataFrame({'loss': rl_losses, 'rewards': rewards})
    loss_df.to_csv(
        learner_combined.model_path + '/results/loss_reward_evolution.csv'
    )
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
            'tox21': [learner_omics.tox21(s) for s in gen_mols_o]
        }
    )
    df.to_csv(os.path.join(learner.model_path, 'results', 'generated.csv'))
    # Plot loss development
    loss_df = pd.DataFrame({'loss': losses_o, 'rewards': rewards_o})
    loss_df.to_csv(
        learner_omics.model_path + '/results/loss_reward_evolution.csv'
    )
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
            'mode': modes_p
        }
    )
    df.to_csv(os.path.join(learner_protein.model_path, 'results', 'generated.csv'))
    # Plot loss development
    loss_df = pd.DataFrame({'loss': losses_p, 'rewards': rewards_p})
    loss_df.to_csv(
        learner_protein.model_path + '/results/loss_reward_evolution.csv'
    )
    plot_loss(
        losses_p,
        rewards_p,
        params['epochs'],
        #protein_name,
        learner_protein.model_path,
        rolling=5
    )