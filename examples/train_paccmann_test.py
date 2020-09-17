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
from paccmann_generator.plot_utils import plot_and_compare, plot_loss
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
model_name = 'test'


disable_rdkit_logging()

params = dict()
params['site'] = 'lung'


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

from paccmann_omics.encoders import ENCODER_FACTORY
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
    protein_params = json.load(f)

protein_predictor = MODEL_FACTORY_PROTEIN['bimodal_mca'](protein_params)
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

PO = ReinforceProteinOmics(generator, protein_encoder, cell_encoder, \
    protein_predictor, paccmann_predictor, protein_df, omics_df, \
    params, generator_smiles_language, model_name, logger)


# Randomly sample a protein
protein_name = np.random.choice(protein_df.index)
print(f'Current train protein: {protein_name}')

# Evaluate on a validation cell line.
train_omics, test_omics = omics_data_splitter(
        omics_df, 'lung', params.get('test_fraction', 0.2)
    )

eval_cell_line = np.random.choice(test_omics)
print(f'Current train cell_line: {eval_cell_line}')

PO.generate_compounds_and_evaluate(1, params['eval_batch_size'], protein_name, eval_cell_line)











from paccmann_chemistry.models import (
    StackGRUDecoder, StackGRUEncoder, TeacherVAE
)
from paccmann_chemistry.utils import get_device
from paccmann_generator import ReinforceOmic
from paccmann_generator.plot_utils import plot_and_compare, plot_loss
from paccmann_generator.utils import add_avg_profile, omics_data_splitter
from paccmann_omics.encoders import ENCODER_FACTORY
from paccmann_predictor.models import MODEL_FACTORY
from pytoda.smiles.smiles_language import SMILESLanguage
from paccmann_generator.utils import disable_rdkit_logging

with open(os.path.join(omics_model_path, 'model_params.json')) as f:
    cell_params = json.load(f)

params = dict()
with open("examples/example_params.json") as f:
    params.update(json.load(f))

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





# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('train_paccmann_rl')
logger_m = logging.getLogger('matplotlib')
logger_m.setLevel(logging.WARNING)

# yapf: disable
parser = argparse.ArgumentParser(description='PaccMann^RL training script')
parser.add_argument(
    'mol_model_path', type=str, help='Path to chemistry model'
)
parser.add_argument(
    'omics_model_path', type=str, help='Path to omics model'
)
parser.add_argument(
    'ic50_model_path', type=str, help='Path to pretrained ic50 model'
)
parser.add_argument(
    'omics_data_path', type=str, help='Omics data path to condition generation'
)
parser.add_argument(
    'params_path', type=str, help='Model params json file directory'
)
parser.add_argument(
    'model_name', type=str, help='Name for the trained model.'
)
parser.add_argument(
    'site', type=str, help='Name of the cancer site for conditioning.'
)


args = parser.parse_args()


# yapf: enable
def main(*, parser_namespace):

    disable_rdkit_logging()

    # read the params json
    params = dict()
    with open(parser_namespace.params_path) as f:
        params.update(json.load(f))

    # get params
    mol_model_path = params.get(
        'mol_model_path', parser_namespace.mol_model_path
    )
    omics_model_path = params.get(
        'omics_model_path', parser_namespace.omics_model_path
    )
    ic50_model_path = params.get(
        'ic50_model_path', parser_namespace.ic50_model_path
    )
    omics_data_path = params.get(
        'omics_data_path', parser_namespace.omics_data_path
    )
    model_name = params.get(
        'model_name', parser_namespace.model_name
    )   # yapf: disable
    site = params.get(
        'site', parser_namespace.site
    )   # yapf: disable

    params['site'] = site

    logger.info(f'Model with name {model_name} starts.')

    # Load omics profiles for conditional generation,
    # complement with avg per site
    omics_df = pd.read_pickle(omics_data_path)
    omics_df = add_avg_profile(omics_df)

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
    generator._associate_language(generator_smiles_language)

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

    # Restore PaccMann
    with open(os.path.join(ic50_model_path, 'model_params.json')) as f:
        paccmann_params = json.load(f)
    paccmann_predictor = MODEL_FACTORY['mca'](paccmann_params)
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

    # Specifies the baseline model used for comparison
    baseline = ReinforceOmic(
        generator, cell_encoder, paccmann_predictor, omics_df, params,
        'baseline', logger
    )

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
    generator_rl._associate_language(generator_smiles_language)

    cell_encoder_rl = ENCODER_FACTORY['dense'](cell_params)
    cell_encoder_rl.load(
        os.path.join(
            omics_model_path,
            f"weights/best_{params.get('metric', 'both')}_encoder.pt"
        ),
        map_location=get_device()
    )
    cell_encoder_rl.eval()
    model_folder_name = site + '_' + model_name
    learner = ReinforceOmic(
        generator_rl, cell_encoder_rl, paccmann_predictor, omics_df, params,
        model_folder_name, logger
    )

    # Split the samples for conditional generation and initialize training
    train_omics, test_omics = omics_data_splitter(
        omics_df, site, params.get('test_fraction', 0.2)
    )
    rewards, rl_losses = [], []
    gen_mols, gen_cell, gen_ic50, modes = [], [], [], []
    logger.info('Models restored, start training.')

    for epoch in range(1, params['epochs'] + 1):

        for step in range(1, params['steps']):

            # Randomly sample a cell line:
            cell_line = np.random.choice(train_omics)

            rew, loss = learner.policy_gradient(
                cell_line, epoch, params['batch_size']
            )
            print(
                f"Epoch {epoch:d}/{params['epochs']:d}, step {step:d}/"
                f"{params['steps']:d}\t loss={loss:.2f}, rew={rew:.2f}"
            )

            rewards.append(rew.item())
            rl_losses.append(loss)

        # Save model
        learner.save(f'gen_{epoch}.pt', f'enc_{epoch}.pt')

        # Compare baseline and trained model on cell line
        base_smiles, base_preds = baseline.generate_compounds_and_evaluate(
            epoch, params['eval_batch_size'], cell_line
        )
        smiles, preds = learner.generate_compounds_and_evaluate(
            epoch, params['eval_batch_size'], cell_line
        )
        gs = [
            s for i, s in enumerate(smiles)
            if preds[i] < learner.ic50_threshold
        ]
        gp = preds[preds < learner.ic50_threshold]
        for p, s in zip(gp, gs):
            gen_mols.append(s)
            gen_cell.append(cell_line)
            gen_ic50.append(p)
            modes.append('train')

        plot_and_compare(
            base_preds, preds, site, cell_line, epoch, learner.model_path,
            'train', params['eval_batch_size']
        )

        # Evaluate on a validation cell line.
        eval_cell_line = np.random.choice(test_omics)
        base_smiles, base_preds = baseline.generate_compounds_and_evaluate(
            epoch, params['eval_batch_size'], eval_cell_line
        )
        smiles, preds = learner.generate_compounds_and_evaluate(
            epoch, params['eval_batch_size'], eval_cell_line
        )
        plot_and_compare(
            base_preds, preds, site, eval_cell_line, epoch, learner.model_path,
            'test', params['eval_batch_size']
        )
        gs = [
            s for i, s in enumerate(smiles)
            if preds[i] < learner.ic50_threshold
        ]
        gp = preds[preds < learner.ic50_threshold]
        for p, s in zip(gp, gs):
            gen_mols.append(s)
            gen_cell.append(eval_cell_line)
            gen_ic50.append(p)
            modes.append('test')

        inds = np.argsort(preds)
        for i in inds[:5]:
            logger.info(
                f'Epoch {epoch:d}, generated {smiles[i]} against '
                f'{eval_cell_line}.\n Predicted IC50 = {preds[i]}. '
            )

        # Save results (good molecules!) in DF
        df = pd.DataFrame(
            {
                'cell_line': gen_cell,
                'SMILES': gen_mols,
                'IC50': gen_ic50,
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
            cell_line,
            learner.model_path,
            rolling=5,
            site=site
        )


if __name__ == '__main__':
    main(parser_namespace=args)
