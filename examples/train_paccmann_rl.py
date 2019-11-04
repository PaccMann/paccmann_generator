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
from paccmann_generator import REINFORCE
from paccmann_generator.plot_utils import plot_and_compare, plot_loss
from paccmann_generator.utils import add_avg_profile, omics_data_splitter
from paccmann_omics.encoders import ENCODER_FACTORY
from paccmann_predictor.models import MODEL_FACTORY
from pytoda.smiles.smiles_language import SMILESLanguage

# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('train_paccmann_rl')

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
    'smiles_language_path', type=str, help='Path to SMILES language object'
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
    smiles_language_path = params.get(
        'smiles_language_path', parser_namespace.smiles_language_path
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

    logger.info(f'Model with name {model_name} starts.')

    # Load SMILES language
    smiles_language = SMILESLanguage.load(smiles_language_path)

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
    generator.load_model(
        os.path.join(
            mol_model_path,
            f"weights/best_{params.get('smiles_metric', 'rec')}.pt"
        ),
        map_location=get_device()
    )

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
    predictor = MODEL_FACTORY['mca'](paccmann_params)
    predictor.load(
        os.path.join(
            ic50_model_path,
            f"weights/best_{params.get('ic50_metric', 'rmse')}_mca.pt"
        ),
        map_location=get_device()
    )
    predictor.eval()

    # Specifies the dumb model used for comparison
    DUMB = REINFORCE(
        generator, cell_encoder, predictor, omics_df, smiles_language, {},
        'dumb', logger
    )

    # Create a fresh model that will be optimized
    gru_encoder_rl = StackGRUEncoder(mol_params)
    gru_decoder_rl = StackGRUDecoder(mol_params)
    generator_rl = TeacherVAE(gru_encoder_rl, gru_decoder_rl)
    generator_rl.load_model(
        os.path.join(
            mol_model_path, f"weights/best_{params.get('metric', 'rec')}.pt"
        ),
        map_location=get_device()
    )
    generator_rl.eval()

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
    LEARNER = REINFORCE(
        generator_rl, cell_encoder_rl, predictor, omics_df, smiles_language,
        params, model_folder_name, logger
    )

    # Split the samples for conditional generation and initialize training
    train_omics, test_omics = omics_data_splitter(
        omics_df, site, params.get('test_fraction', 0.2)
    )
    rewards, rl_losses = [], []
    gen_mols, gen_cell, gen_ic50, tt = [], [], [], []

    for epoch in range(1, params['epochs'] + 1):

        for step in range(1, params['steps']):

            # Randomly sample a cell line:
            cell_line = np.random.choice(train_omics)

            rew, loss = LEARNER.policy_gradient(
                cell_line, epoch, params['batch_size']
            )
            print(
                f"Epoch {epoch:d}/{params['epochs']:d}, step {step:d}/"
                f"{params['steps']:d}\t loss={loss:.2f}, rew={rew:.2f}"
            )

            rewards.append(rew.item())
            rl_losses.append(loss)

        # Save model
        LEARNER.save(f'gen_{epoch}.pt', f'enc_{epoch}.pt')

        # Compare dumb and trained model on cell line
        unbiased_smiles, unbiased_preds = DUMB.generate_compounds_and_evaluate(
            epoch, params['eval_batch_size'], cell_line
        )
        smiles, preds = LEARNER.generate_compounds_and_evaluate(
            epoch, params['eval_batch_size'], cell_line
        )
        gs = [
            s for i, s in enumerate(smiles)
            if preds[i] < LEARNER.ic50_threshold
        ]
        gp = preds[preds < LEARNER.ic50_threshold]
        for p, s in zip(gp, gs):
            gen_mols.append(s)
            gen_cell.append(cell_line)
            gen_ic50.append(p)
            tt.append('train')

        plot_and_compare(
            unbiased_preds, preds, site, cell_line, epoch, LEARNER.model_path,
            'train', params['eval_batch_size']
        )

        # Evaluate on a validation cell line.
        eval_cell_line = np.random.choice(test_omics)
        unbiased_smiles, unbiased_preds = DUMB.generate_compounds_and_evaluate(
            epoch, params['eval_batch_size'], eval_cell_line
        )
        smiles, preds = LEARNER.generate_compounds_and_evaluate(
            epoch, params['eval_batch_size'], eval_cell_line
        )
        plot_and_compare(
            unbiased_preds, preds, site, eval_cell_line, epoch,
            LEARNER.model_path, 'test', params['eval_batch_size']
        )
        gs = [
            s for i, s in enumerate(smiles)
            if preds[i] < LEARNER.ic50_threshold
        ]
        gp = preds[preds < LEARNER.ic50_threshold]
        for p, s in zip(gp, gs):
            gen_mols.append(s)
            gen_cell.append(eval_cell_line)
            gen_ic50.append(p)
            tt.append('test')

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
                'mode': tt
            }
        )
        df.to_csv(os.path.join(LEARNER.model_path, 'results', 'generated.csv'))
        # Plot loss development
        loss_df = pd.DataFrame({'loss': rl_losses, 'rewards': rewards})
        loss_df.to_csv(
            LEARNER.model_path + '/results/loss_reward_evolution.csv'
        )
        plot_loss(
            rl_losses,
            rewards,
            params['epochs'],
            cell_line,
            LEARNER.model_path,
            rolling=5,
            site=site
        )


if __name__ == '__main__':
    main(parser_namespace=args)
