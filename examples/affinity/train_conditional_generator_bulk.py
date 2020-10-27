"""Script to test a conditional generator on a bulk of molecules"""
import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
from paccmann_omics.encoders import ENCODER_FACTORY

from paccmann_chemistry.models import StackGRUDecoder, StackGRUEncoder, TeacherVAE
from paccmann_chemistry.utils import get_device
from paccmann_generator import ReinforceProtein
from paccmann_generator.plot_utils import plot_and_compare_proteins, plot_loss
from paccmann_generator.utils import disable_rdkit_logging
from paccmann_predictor.models import MODEL_FACTORY
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
    'train_protein_path', type=str, help='Path to protein data for conditioning'
)
parser.add_argument(
    'test_protein_path', type=str, help='Path to protein data to evaluate generator'
)
parser.add_argument(
    'params_path', type=str, help='Model params json file directory'
)
parser.add_argument(
    'unbiased_predictions_path', type=str,
    help='Path to folder with aff. preds for 3000 mols from unbiased generator. '
    'For each protein in train/test file, there should be one .csv'
)
parser.add_argument(
    'model_name', type=str, help='Name for the trained model.'
)
parser.add_argument(
    '--tox21_path', help='Optional path to Tox21 model.'
)
parser.add_argument(
    '--organdb_path', help='Optional path to OrganDB model.'
)
parser.add_argument(
    '--site', help='Specify a site in case of using a OrganDB model.'
)
parser.add_argument(
    '--clintox_path', help='Optional path to ClinTox model.'
)
parser.add_argument(
    '--sider_path', help='Optional path to SIDER model.'
)

args = parser.parse_args()


# yapf: enable
def main(*, parser_namespace):

    disable_rdkit_logging()

    # read the params json
    params = dict()
    with open(parser_namespace.params_path) as f:
        params.update(json.load(f))

    # get params, json args take precedence
    mol_model_path = params.get(
        'mol_model_path', parser_namespace.mol_model_path
    )
    protein_model_path = params.get(
        'protein_model_path', parser_namespace.protein_model_path
    )
    affinity_model_path = params.get(
        'affinity_model_path', parser_namespace.affinity_model_path
    )
    train_protein_path = params.get(
        'train_protein_path', parser_namespace.train_protein_path
    )
    test_protein_path = params.get(
        'test_protein_path', parser_namespace.test_protein_path
    )
    model_name = params.get('model_name', parser_namespace.model_name)
    unbiased_preds_path = params.get(
        'unbiased_predictions_path', parser_namespace.unbiased_predictions_path
    )
    logger.info(f'Model with name {model_name} starts.')

    # passing optional paths to params to possibly update_reward_fn
    optional_reward_args = [
        'tox21_path',
        'organdb_path',
        'site',
        'clintox_path',
        'sider_path',
    ]
    for arg in optional_reward_args:
        if parser_namespace.__dict__[arg]:
            # json still has precedence
            params[arg] = params.get(arg, parser_namespace.__dict__[arg])

    # Load protein sequence data
    test_protein_df = pd.read_csv(test_protein_path, index_col=0)
    train_protein_df = pd.read_csv(train_protein_path, index_col=0)
    protein_df = (
        pd.concat([train_protein_df, test_protein_df]).drop_duplicates()
    )
    # Drop duplicate entries
    protein_df = protein_df[~protein_df.index.duplicated()]

    logger.info(f'Test proteins are {list(test_protein_df.index)}')

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
        map_location=get_device(),
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
            f"weights/best_{params.get('omics_metric','both')}_encoder.pt",
        ),
        map_location=get_device(),
    )
    protein_encoder.eval()

    # Restore affinity predictor
    with open(os.path.join(affinity_model_path, 'model_params.json')) as f:
        predictor_params = json.load(f)
    predictor = MODEL_FACTORY['bimodal_mca'](predictor_params)
    predictor.load(
        os.path.join(
            affinity_model_path,
            f"weights/best_{params.get('p_metric', 'ROC-AUC')}_bimodal_mca.pt",
        ),
        map_location=get_device(),
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

    # Create a fresh model that will be optimized
    gru_encoder_rl = StackGRUEncoder(mol_params)
    gru_decoder_rl = StackGRUDecoder(mol_params)
    generator_rl = TeacherVAE(gru_encoder_rl, gru_decoder_rl)
    generator_rl.load(
        os.path.join(
            mol_model_path, f"weights/best_{params.get('metric', 'rec')}.pt"
        ),
        map_location=get_device(),
    )
    generator_rl.eval()
    # Load languages
    generator_rl._associate_language(generator_smiles_language)

    protein_encoder_rl = ENCODER_FACTORY['dense'](protein_params)
    protein_encoder_rl.load(
        os.path.join(
            protein_model_path,
            f"weights/best_{params.get('metric', 'both')}_encoder.pt",
        ),
        map_location=get_device(),
    )
    protein_encoder_rl.eval()
    model_folder_name = model_name
    learner = ReinforceProtein(
        generator_rl,
        protein_encoder_rl,
        predictor,
        protein_df,
        params,
        model_folder_name,
        logger,
    )
    logger.info('Models restored')

    # Specifies the baseline model used for comparison
    unbiased_preds = pd.concat(
        [
            pd.read_csv(
                os.path.join(unbiased_preds_path, protein + '.csv'),
                names=['SMILES', protein],
                header=0,
            ) for protein in list(train_protein_df.index) +
            list(test_protein_df.index)
        ],
        join='inner',
        axis=1
    ).drop_duplicates().T.drop_duplicates().T
    logger.info('Baseline predictions restored.')

    # Every protein x epoch x 2 (affinity + toxicity)
    biased_ratios = np.zeros((len(test_protein_df), params['epochs'], 2))
    rewards, rl_losses = [], []

    logger.info(f'Model stored at {learner.model_path}')

    for epoch in range(1, params['epochs'] + 1):

        for step in range(1, params['steps']):

            # Randomly sample a protein
            protein_name = np.random.choice(protein_df.index)
            while protein_name in test_protein_df.index:
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

        # Generate for each protein
        for idx, test_protein in enumerate(test_protein_df.index):

            smiles, preds = learner.generate_compounds_and_evaluate(
                epoch, params['eval_batch_size'], test_protein
            )

            # Get toxicity
            toxes = [learner.tox21(s) for s in smiles]

            # Filter
            idxs = [i for i, s in enumerate(smiles) if preds[i] > 0.5]

            if len(idxs) > 0:
                pd.DataFrame(
                    {
                        'protein': test_protein,
                        'SMILES': list(np.array(smiles)[idxs]),
                        'Binding probability': list(np.array(preds)[idxs]),
                        'Tox21': list(np.array(toxes)[idxs]),
                    }
                ).to_csv(
                    os.path.join(
                        learner.model_path, 'results', 'generated.csv'
                    )
                )

            inds = np.argsort(preds)[::-1]
            for i in inds[:5]:
                logger.info(
                    f'Epoch {epoch:d}, generated {smiles[i]} against '
                    f'{test_protein}.\n Predicted IC50 = {preds[i]}. '
                )
            plot_and_compare_proteins(
                unbiased_preds[test_protein],
                preds,
                test_protein,
                epoch,
                learner.model_path,
                'eval',
                params['eval_batch_size'],
            )

            efficay_ratio = round(100 * np.sum(preds > 0.5) / len(preds), 2)
            toxicity_ratio = round(100 * np.sum(toxes == 1.0) / len(toxes), 2)

            biased_ratios[idx, epoch - 1, 0] = efficay_ratio
            biased_ratios[idx, epoch - 1, 1] = toxicity_ratio

        logger.info(f'Fraction of binding compounds: {efficay_ratio}')
        logger.info(f'Fraction of non-toxic: {toxicity_ratio}')
        pd.DataFrame(
            {
                'epoch': epoch,
                'protein': test_protein,
                'efficacy_ratio': efficay_ratio,
                'tox_ratio': toxicity_ratio
            },
            index=[epoch]
        ).to_csv(learner.model_path + '/results/ratios.csv', mode='a')

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
            rolling=5,
        )


if __name__ == '__main__':
    main(parser_namespace=args)
