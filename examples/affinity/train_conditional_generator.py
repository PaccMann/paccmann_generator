import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
from paccmann_omics.encoders import ENCODER_FACTORY

from paccmann_chemistry.models import (
    StackGRUDecoder, StackGRUEncoder, TeacherVAE
)
from paccmann_chemistry.utils import get_device
from paccmann_generator import ReinforceProtein
from paccmann_generator.plot_utils import plot_and_compare_proteins, plot_loss
from paccmann_generator.utils import disable_rdkit_logging
from paccmann_predictor.models.predictors import MODEL_FACTORY
from pytoda.files import read_smi
from pytoda.proteins.protein_language import ProteinLanguage
from pytoda.smiles.smiles_language import SMILESLanguage

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
    'protein_model_path', type=str, help='Path to protein model'
)
parser.add_argument(
    'affinity_model_path', type=str, help='Path to pretrained affinity model'
)
parser.add_argument(
    'protein_data_path', type=str, help='Path to protein data for conditioning'
)
parser.add_argument(
    'params_path', type=str, help='Model params json file directory'
)
parser.add_argument(
    'model_name', type=str, help='Name for the trained model.'
)
parser.add_argument(
    'test_protein_id', type=str, help='ID of testing protein (LOOCV).'
)
parser.add_argument(
    'unbiased_predictions_path', type=str,
    help='Path to folder with aff. preds for 3000 mols from unbiased generator'
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
    protein_model_path = params.get(
        'protein_model_path', parser_namespace.protein_model_path
    )
    affinity_model_path = params.get(
        'affinity_model_path', parser_namespace.affinity_model_path
    )
    protein_data_path = params.get(
        'protein_data_path', parser_namespace.protein_data_path
    )
    model_name = params.get(
        'model_name', parser_namespace.model_name
    )   # yapf: disable
    test_id = int(params.get(
        'test_protein_id', parser_namespace.test_protein_id
    ))   # yapf: disable
    unbiased_preds_path = params.get(
        'unbiased_predictions_path', parser_namespace.unbiased_predictions_path
    )   # yapf: disable
    model_name += '_' + str(test_id)
    logger.info(f'Model with name {model_name} starts.')

    # Load protein sequence data
    if protein_data_path.endswith('.smi'):
        protein_df = read_smi(protein_data_path, names=['Sequence'])
    elif protein_data_path.endswith('.csv'):
        protein_df = pd.read_csv(protein_data_path, index_col='entry_name')
    else:
        raise TypeError(
            f"{protein_data_path.split('.')[-1]} files are not supported."
        )

    protein_test_name = protein_df.iloc[test_id].name
    logger.info(f'Test protein is {protein_test_name}')

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

    # Restore affinity predictor
    with open(os.path.join(affinity_model_path, 'model_params.json')) as f:
        predictor_params = json.load(f)
    predictor = MODEL_FACTORY['bimodal_mca'](predictor_params)
    predictor.load(
        os.path.join(
            affinity_model_path,
            f"weights/best_{params.get('p_metric', 'ROC-AUC')}_bimodal_mca.pt"
        ),
        map_location=get_device()
    )
    predictor.eval()

    # Load languages
    affinity_smiles_language = SMILESLanguage.load(
        os.path.join(affinity_model_path, 'smiles_language.pkl')
    )
    affinity_protein_language = ProteinLanguage.load(
        os.path.join(affinity_model_path, 'protein_language.pkl')
    )
    predictor._associate_language(affinity_smiles_language)
    predictor._associate_language(affinity_protein_language)

    # Specifies the baseline model used for comparison
    unbiased_preds = np.array(
        pd.read_csv(
            os.path.join(unbiased_preds_path, protein_test_name + '.csv')
        )['affinity'].values
    )  # yapf: disable

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
    # Load languages
    generator_rl._associate_language(generator_smiles_language)

    protein_encoder_rl = ENCODER_FACTORY['dense'](protein_params)
    protein_encoder_rl.load(
        os.path.join(
            protein_model_path,
            f"weights/best_{params.get('metric', 'both')}_encoder.pt"
        ),
        map_location=get_device()
    )
    protein_encoder_rl.eval()
    model_folder_name = model_name
    learner = ReinforceProtein(
        generator_rl, protein_encoder_rl, predictor, protein_df, params,
        model_folder_name, logger
    )

    biased_ratios, tox_ratios = [], []
    rewards, rl_losses = [], []
    gen_mols, gen_prot, gen_affinity, mode = [], [], [], []

    logger.info(f'Model stored at {learner.model_path}')

    for epoch in range(1, params['epochs'] + 1):

        for step in range(1, params['steps']):

            # Randomly sample a protein
            protein_name = np.random.choice(protein_df.index)
            while protein_name == protein_test_name:
                protein_name = np.random.choice(protein_df.index)

            logger.info(f'Current train protein: {protein_name}')

            rew, loss = learner.policy_gradient(
                protein_name, epoch, params['batch_size']
            )
            logger.info(
                f"Epoch {epoch:d}/{params['epochs']:d}, step {step:d}/"
                f"{params['steps']:d}\t loss={loss:.2f}, mean rew={rew:.2f}"
            )

            rewards.append(rew.item())
            rl_losses.append(loss)

        # Save model
        if epoch % 10 == 0:
            learner.save(f'gen_{epoch}.pt', f'enc_{epoch}.pt')
        logger.info(f'EVAL protein: {protein_test_name}')

        smiles, preds = (
            learner.generate_compounds_and_evaluate(
                epoch, params['eval_batch_size'], protein_test_name
            )
        )
        gs = [s for i, s in enumerate(smiles) if preds[i] > 0.5]
        gp = preds[preds > 0.5]
        for p, s in zip(gp, gs):
            gen_mols.append(s)
            gen_prot.append(protein_test_name)
            gen_affinity.append(p)
            mode.append('eval')

        inds = np.argsort(gp)[::-1]
        for i in inds[:5]:
            logger.info(
                f'Epoch {epoch:d}, generated {gs[i]} against '
                f'{protein_test_name}.\n Predicted IC50 = {gp[i]}. '
            )

        plot_and_compare_proteins(
            unbiased_preds, preds, protein_test_name, epoch,
            learner.model_path, 'train', params['eval_batch_size']
        )
        biased_ratios.append(
            np.round(100 * (np.sum(preds > 0.5) / len(preds)), 1)
        )
        all_toxes = np.array([learner.tox21(s) for s in smiles])
        tox_ratios.append(
            np.round(100 * (np.sum(all_toxes == 1.) / len(all_toxes)), 1)
        )
        logger.info(f'Percentage of non-toxic compounds {tox_ratios[-1]}')

        toxes = [learner.tox21(s) for s in gen_mols]
        # Save results (good molecules!) in DF
        df = pd.DataFrame(
            {
                'protein': gen_prot,
                'SMILES': gen_mols,
                'Binding probability': gen_affinity,
                'mode': mode,
                'Tox21': toxes
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
            protein_name,
            learner.model_path,
            rolling=5
        )
    pd.DataFrame({
        'efficacy_ratio': biased_ratios,
        'tox_ratio': tox_ratios
    }).to_csv(learner.model_path + '/results/ratios.csv')


if __name__ == '__main__':
    main(parser_namespace=args)
