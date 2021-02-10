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

with open(params_path) as f:
    params.update(json.load(f))

# Load omics profiles for conditional generation,
# complement with avg per site
omics_df = pd.read_pickle(omics_data_path)
omics_df = add_avg_profile(omics_df)
idx = [i in cancer_cell_lines for i in omics_df['cell_line']]
omics_df  = omics_df[idx]
#omics_df = omics_df[omics_df.histology == cancertype]
print("omics data:", omics_df.shape, omics_df['cell_line'].iloc[0])
test_cell_line = omics_df['cell_line'].iloc[0]
print("test_cell_line:", test_cell_line)
model_name = model_name + '_'+ test_cell_line + '_lern' + str(params['learning_rate'])+'_aromaticity' + str(params['aromaticity_weight'])
model_name ='test_test'
logger.info(f'Model with name {model_name} starts.')

# Load protein sequence data
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

# Specifies the baseline model used for comparison
unbiased_preds_df = pd.read_csv(unbiased_predictions_path)

def create_combined_model(ensemble_type):
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


    #remove_invalid = remove_invalid

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

    model_folder_name = site + '_' + ensemble_type + '_' + model_name + '_combined'
    combined = ReinforceProteinOmics(
        generator_rl, protein_encoder_rl,cell_encoder_rl, \
        protein_predictor, paccmann_predictor, protein_df, omics_df, \
        params, generator_smiles_language, model_folder_name, logger, remove_invalid, ensemble_type=ensemble_type
    )
    return combined

def save_loss_reward(losses, rewards, smiles_steps, epoch, learner, protein_steps=None, cells=None):
    #Plot loss development
    if(protein_steps is None):
        loss_df = pd.DataFrame({'loss': losses, 'rewards': rewards, 'smiles':smiles_steps, 'cell_line':cells, 'epoch':epoch})
    elif(cells is None):
        loss_df = pd.DataFrame({'loss': losses, 'rewards': rewards, 'smiles':smiles_steps, 'proteins':protein_steps, 'epoch':epoch})
    else:
        loss_df = pd.DataFrame({'loss': losses, 'rewards': rewards, 'smiles':smiles_steps, 'proteins':protein_steps, 'cell_line':cells,  'epoch':epoch})
    if epoch ==1:
        loss_df.to_csv(learner.model_path + '/results/loss_reward_evolution.csv')
    else:
        loss_df.to_csv(learner.model_path + '/results/loss_reward_evolution.csv', mode='a', header=False)
    losses_rewards_all = pd.read_csv(learner.model_path + '/results/loss_reward_evolution.csv', header = 0)
    rewards_all = losses_rewards_all['rewards']
    losses_all = losses_rewards_all['loss']
    plot_loss(
        losses_all,
        rewards_all,
        params['epochs'],
        #protein_name,
        learner.model_path,
        rolling=5
    )
    
def train_and_save_steps(learner, epoch, params, smiles_steps, rewards, losses, protein_steps=None, cells=None, protein_name = None, cell_line = None):
    if (protein_name is None):
        rew, loss, smiles_step, smiles_idx_steps = learner.policy_gradient(
            cell_line, epoch, params
        )
        cell = (cell_line*params)[:params]
        cell = [cell[idx] for idx in smiles_idx_steps]
        cells.append(cell)
    elif (cell_line is None):
        rew, loss, smiles_step, smiles_idx_steps = learner.policy_gradient(
            protein_name, epoch, params
        )
        proteins = (protein_name*params)[:params]
        proteins = [proteins[idx] for idx in smiles_idx_steps]
        protein_steps.append(proteins)
    else:
        #### TO-DO change policy_gradient function of combined model to be similar to single ones.
        rew, loss, smiles_step, smiles_idx_steps = learner.policy_gradient(
            protein_name, cell_line, epoch, params
        )
        proteins = (protein_name*params)[:params]
        proteins = [proteins[idx] for idx in smiles_idx_steps]
        protein_steps.append(proteins)
        cell = (cell_line*params)[:params]
        cell = [cell[idx] for idx in smiles_idx_steps]
        cells.append(cell)

    print(f"Epoch {epoch:d}")
    rewards.append(rew.item())
    losses.append(loss)
    smiles_steps.append(smiles_step)
    return losses, rewards, smiles_steps, protein_steps, cells

def generate_and_save(epoch, learner, mode, param, gen_mols, unbiased_predsP=None, unbiased_predsO=None, protein_name=None, cell_line=None, prot=None, aff=None, cell=None, ic50=None, modes=[]):
    dict_res = {}
    if (cell_line is None):
        smiles, predsP, idx = learner.generate_compounds_and_evaluate(
            epoch, param, protein_name
        )
    elif(protein_name is None):
        smiles, predsO, idx = learner.generate_compounds_and_evaluate(
            epoch, param, cell_line
        )
    else:
        smiles, predsP, predsO, idx = learner.generate_compounds_and_evaluate(
            epoch, param, protein_name, cell_line
        )
    
    if(cell_line is not None):
        cells = (cell_line*param)[:param]
        cells = [cells[i] for i in idx]
        cells = [o for i,o in enumerate(cells) if (predsO[i] < learner.ic50_threshold)]
    if(protein_name is not None):
        proteins = (protein_name*param)[:param]
        proteins = [proteins[i] for i in idx]
        proteins = [p for i, p in enumerate(proteins) if predsP[i] > 0.5]

    if (cell_line is None):
        gs = [s for i, s in enumerate(smiles) if predsP[i] > 0.5]
        gp_p = preds[predsP > 0.5]
        for p_p, p in zip(gp_p, proteins):
            prot.append(p)
            aff.append(p_p)
        dict_res = {
            'protein': prot,
            'Binding probability': aff
        }
        plot_and_compare_proteins(
            unbiased_predsP, predsP, protein_name, epoch, learner.model_path,
            mode, param
        )
    elif(protein_name is None):
        gs = [
            s for i, s in enumerate(smiles)
            if predsO[i] < learner.ic50_threshold
        ]
        gp_o = predsO[predsO < learner.ic50_threshold]
        for p_o, c in zip(gp_o, cells):
            cell.append(c)
            ic50.append(p_o)
        dict_res = {
            'cell_line': cell,
            'IC50': ic50
        }
        plot_and_compare(
            unbiased_predsO, predsO, site, cell_line, epoch, learner.model_path,
            mode, param
        )
    else:
        gs = [
            s for i, s in enumerate(smiles)
            if predsO[i] < learner.ic50_threshold and predsP[i] > 0.5
        ]
        gp_o = predsO[(predsO < learner.ic50_threshold) & (predsP > 0.5)]
        gp_p = predsP[(predsO < learner.ic50_threshold) & (predsP > 0.5)]
        for p_o, p_p, p, c in zip(gp_o, gp_p, proteins, cells):
            cell.append(c)
            prot.append(p)
            aff.append(p_p)
            ic50.append(p_o)
        dict_res = {
            'protein': prot,
            'Binding probability': aff,
            'cell_line': cell,
            'IC50': ic50
        }
        plot_and_compare(
            unbiased_predsO, predsO, site, cell_line, epoch, learner.model_path,
            mode, param
        )
        plot_and_compare_proteins(
            unbiased_predsP, predsP, protein_name, epoch, learner.model_path,
            mode, param
        )

    for s in gs:
        gen_mols.append(s)
        modes.append(mode)
    dict_res['SMILES'] = gen_mols
    dict_res['mode'] = modes
    dict_res['epoch'] = [epoch] * len(gen_mols)

    
    if mode == 'test':
        df = pd.DataFrame(dict_res)
        if epoch ==1:
            df.to_csv(os.path.join(learner.model_path, 'results', 'generated.csv'))
        else:
            df.to_csv(os.path.join(learner.model_path, 'results', 'generated.csv'), mode='a', header=False)

    return gen_mols, prot, aff, cell, ic50, modes
    
concat_combined = create_combined_model('concat')
average_combined = create_combined_model('average')

# Split the samples for conditional generation and initialize training
train_omics, test_omics = omics_data_splitter(
    omics_df, site, params.get('test_fraction', 0.2)
)
#train_protein, test_protein = protein_data_splitter(
#    protein_df, params.get('test_fraction', 0.2)
#)
train_protein = protein_df
test_protein = protein_df
print("test omics:", len(test_omics), ":test protein", test_protein.shape, ":train omics:", len(train_omics), ":train protein:", train_protein.shape)

logger.info('Models restored, start training.')

# choose a validation cell line and protein.
#eval_cell_lines = np.random.choice(test_omics, size = 20, replace=True)
#eval_cell_lines = [str(i) for i in eval_cell_lines]
eval_cell_lines = [test_cell_line]
eval_protein_names = np.random.choice(test_protein.index, size = 20, replace=False)
eval_protein_names = [str(i) for i in eval_protein_names]

for epoch in range(1, params['epochs'] + 1):
    rewards_average, losses_average, rewards_concat, losses_concat = [], [], [], []
    cell_steps_average, protein_steps_average, smiles_steps_average = [], [], []
    cell_steps_concat, protein_steps_concat, smiles_steps_concat = [], [], []
    gen_mols_average ,gen_prot_average, gen_affinity_average, gen_cell_average, gen_ic50_average, modes_average = [], [], [], [], [], []
    gen_mols_concat ,gen_prot_concat, gen_affinity_concat, gen_cell_concat, gen_ic50_concat, modes_concat = [], [], [], [], [], []
    
    for step in range(1, params['steps'] + 1):

        # Randomly sample a cell lines and proteins for training:
        #cell_line = np.random.choice(train_omics, size = 20, replace=True)
        #cell_line = [str(i) for i in cell_line]
        cell_line  = [i for i in omics_df['cell_line'] if i != test_cell_line]
        protein_name = np.random.choice(train_protein.index, size = 20, replace=False)
        protein_name = [str(i) for i in protein_name]
        print(f'Current train cell_lines: {cell_line}')
        # Randomly sample a protein
        #protein_name = [np.random.choice(train_protein.index)]
        print(f'Current train proteins: {protein_name}')
        
        losses_average, rewards_average, smiles_steps_average, protein_steps_average, cell_steps_average = train_and_save_steps(
            average_combined, epoch, params['batch_size'], smiles_steps_average, rewards_average, losses_average, 
            protein_steps=protein_steps_average, protein_name=protein_name, cells=cell_steps_average, cell_line=cell_line)
        losses_concat, rewards_concat, smiles_steps_concat, protein_steps_concat, cell_steps_concat = train_and_save_steps(
            concat_combined, epoch, params['batch_size'], smiles_steps_concat, rewards_concat, losses_concat, 
            protein_steps=protein_steps_concat, protein_name=protein_name, cells=cell_steps_concat, cell_line=cell_line)

    # Save model
    average_combined.save(f'gen_{epoch}.pt', f'enc_{epoch}_protein.pt', f'enc_{epoch}_omics.pt')
    concat_combined.save(f'gen_{epoch}.pt', f'enc_{epoch}_protein.pt', f'enc_{epoch}_omics.pt')
    unbiased_predsP = unbiased_preds_df[protein_name].values.reshape(-1)[:params['batch_size']]
    unbiased_predsO = unbiased_preds_df[cell_line].values.reshape(-1)[:params['batch_size']]
    save_loss_reward(losses_average, rewards_average, smiles_steps_average, epoch, average_combined, 
        protein_steps=protein_steps_average, cells=cell_steps_average)
    save_loss_reward(losses_concat, rewards_concat, smiles_steps_concat, epoch, concat_combined, 
        protein_steps=protein_steps_concat, cells=cell_steps_concat)
    
    gen_mols_average, gen_prot_average, gen_affinity_average, gen_cell_average, gen_ic50_average, modes_average = generate_and_save(
        epoch, average_combined, 'train', params['batch_size'],
        gen_mols_average, unbiased_predsP=unbiased_predsP, protein_name=protein_name, prot=gen_prot_average, aff=gen_affinity_average, 
        unbiased_predsO=unbiased_predsO, cell_line=cell_line, cell=gen_cell_average, ic50=gen_ic50_average, modes=modes_average)
    gen_mols_concat, gen_prot_concat, gen_affinity_concat, gen_cell_concat, gen_ic50_concat, modes_concat = generate_and_save(
        epoch, concat_combined, 'train', params['batch_size'],
        gen_mols_concat, unbiased_predsP=unbiased_predsP, protein_name=protein_name, prot=gen_prot_concat, aff=gen_affinity_concat, 
        unbiased_predsO=unbiased_predsO, cell_line=cell_line, cell=gen_cell_concat, ic50=gen_ic50_concat, modes=modes_concat)
    
    # Evaluate on a validation cell line and protein.
    unbiased_predsP = unbiased_preds_df[eval_protein_names].values.reshape(-1)[:params['eval_batch_size']]
    unbiased_predsO = unbiased_preds_df[eval_cell_lines].values.reshape(-1)[:params['eval_batch_size']]
    
    gen_mols_average, gen_prot_average, gen_affinity_average, gen_cell_average, gen_ic50_average, modes_average = generate_and_save(
        epoch, average_combined, 'test', params['eval_batch_size'],
        gen_mols_average, unbiased_predsP=unbiased_predsP, protein_name=eval_protein_names, prot=gen_prot_average, aff=gen_affinity_average, 
        unbiased_predsO=unbiased_predsO, cell_line=eval_cell_lines, cell=gen_cell_average, ic50=gen_ic50_average, modes=modes_average)
    gen_mols_concat, gen_prot_concat, gen_affinity_concat, gen_cell_concat, gen_ic50_concat, modes_concat = generate_and_save(
        epoch, concat_combined, 'test', params['eval_batch_size'],
        gen_mols_concat, unbiased_predsP=unbiased_predsP, protein_name=eval_protein_names, prot=gen_prot_concat, aff=gen_affinity_concat, 
        unbiased_predsO=unbiased_predsO, cell_line=eval_cell_lines, cell=gen_cell_concat, ic50=gen_ic50_concat, modes=modes_concat)