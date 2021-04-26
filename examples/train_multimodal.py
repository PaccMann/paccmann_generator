import argparse
import json
import logging
import os
import sys
import numpy as np
import pandas as pd
import warnings
import torch
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
from paccmann_sets.models.sets_autoencoder import SetsAE
import sys
sys.path.append('/home/tol/paccmann_affinity')
from paccmann_affinity.models.predictors import MODEL_FACTORY as MODEL_FACTORY_PROTEIN

from paccmann_generator.model import Model
from paccmann_generator.reinforce_proteins_omics import ReinforceProteinOmics
from paccmann_generator import ReinforceOmic
from paccmann_generator.reinforce_proteins import ReinforceProtein

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
    'params_path', type=str, help='Model params json file directory'
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
    'params_omics_path', type=str, help='Omics model params json file directory'
)
parser.add_argument(
    'test_cell_line', type=str, help='name of the test cell line'
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
    'protein_data_seq_path', type=str, help='Path to protein sequence data for conditioning'
)
parser.add_argument(
    'params_protein_path', type=str, help='protein model params json file directory'
)
parser.add_argument(
    'set_encoder_path', type=str, help='Path to set encoder model'
)
parser.add_argument(
    'model_name', type=str, help='Name for the trained model.'
)
parser.add_argument(
    'site', type=str, help='Name of the cancer site for conditioning.'
)
parser.add_argument(
    'cancertype', type=str, help='Name of the cancer type for conditioning.'
)
parser.add_argument(
    'unbiased_path', type=str,
    help='Path to folder with unbiased model'
)
parser.add_argument(
    'remove_invalid', type=bool, 
    help='Sanitizing/removing the invalid smiles during training.'
)
parser.add_argument(
    'cancer_genes',
    help='a list with genes to consider.'
)
parser.add_argument(
    'cancer_cell_lines', 
    help='a list with cell lines to consider.'
)
parser.add_argument(
    '--tox21_path', help='Optional path to Tox21 model.'
)
parser.add_argument(
    '--organdb_path', help='Optional path to OrganDB model.'
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

    params = dict()
    params_o, params_p = dict(), dict()
    with open(parser_namespace.params_path) as f:
        params.update(json.load(f))
    with open(parser_namespace.params_omics_path) as f:
        params_o.update(json.load(f))
    with open(parser_namespace.params_protein_path) as f:
        params_p.update(json.load(f))

    # get params
    mol_model_path = params.get(
        'mol_model_path', parser_namespace.mol_model_path
    )
    params['mol_model_path']=mol_model_path
    omics_model_path = params.get(
        'omics_model_path', parser_namespace.omics_model_path
    )
    params['omics_model_path']=omics_model_path
    ic50_model_path = params.get(
        'ic50_model_path', parser_namespace.ic50_model_path
    )
    params['ic50_model_path']=ic50_model_path
    omics_data_path = params.get(
        'omics_data_path', parser_namespace.omics_data_path
    )
    model_name = params.get(
        'model_name', parser_namespace.model_name
    )   # yapf: disable
    params['model_name']=model_name
    site = params.get(
        'site', parser_namespace.site
    )   # yapf: disable# get params, json args take precedence
    protein_model_path = params.get(
        'protein_model_path', parser_namespace.protein_model_path
    )
    params['protein_model_path']=protein_model_path
    affinity_model_path = params.get(
        'affinity_model_path', parser_namespace.affinity_model_path
    )
    params['affinity_model_path']=affinity_model_path
    protein_data_path = params.get(
        'protein_data_path', parser_namespace.protein_data_path
    )
    protein_data_seq_path = params.get(
        'protein_data_seq_path', parser_namespace.protein_data_seq_path
    )
    set_encoder_path = params.get(
        'set_encoder_path', parser_namespace.set_encoder_path
    )
    params['set_encoder_path']=set_encoder_path
    model_name = params.get(
        'cancertype', parser_namespace.cancertype
    )   # yapf: disable
    test_cell_line = params.get(
        'test_cell_line', parser_namespace.test_cell_line
    )   # yapf: disable
    unbiased_predictions_path = params.get(
        'unbiased_predictions_path', parser_namespace.unbiased_path
    )   # yapf: disable
    remove_invalid = params.get(
        'remove_invalid', parser_namespace.remove_invalid
    )   # yapf: disable
    params['remove_invalid']=remove_invalid
    cancer_genes = params.get(
        'cancer_genes', parser_namespace.cancer_genes
    )   # yapf: disable
    cancer_cell_lines = params.get(
        'cancer_cell_lines', parser_namespace.cancer_cell_lines
    )   # yapf: disable
    cancertype = params.get(
        'cancertype', parser_namespace.cancertype
    )   # yapf: disable

    params['site']=site
    params_o['site']=site
    params_p['site']=site
    params['cancertype'] = cancertype

    # Load omics profiles for conditional generation,
    # complement with avg per site
    omics_df = pd.read_pickle(omics_data_path)
    omics_df = add_avg_profile(omics_df)
    idx = [i in cancer_cell_lines for i in omics_df['cell_line']]
    omics_df  = omics_df[idx]
    #omics_df = omics_df[omics_df.histology == cancertype]
    print("omics data:", omics_df.shape)
    print("test_cell_line:", test_cell_line)
    model_name = model_name + '_'+ test_cell_line + '_lern' + str(params['learning_rate'])
    logger.info(f'Model with name {model_name} starts.')

    # Load protein sequence data
    protein_df = pd.read_csv(protein_data_path, index_col=0)
    protein_df = protein_df[~protein_df.index.isnull()]
    protein_df.index = [i.split('|')[2] for i in protein_df.index]
    protein_seq_df = pd.read_csv(protein_data_seq_path, names = ['sequence'], index_col=0)
    protein_seq_df.index = [i.split('|')[2] for i in protein_seq_df.index]
    protein_df = pd.concat([protein_df, protein_seq_df], axis=1, join='outer')
    protein_df = protein_df[[s.split('_')[0] in cancer_genes for s in protein_df.index]]
    print("proteins:", protein_df.index, len(cancer_genes))

    # Specifies the baseline model used for comparison
    unbiased_preds_df = pd.read_csv(unbiased_predictions_path)

    model_types = ['onlyConcat', 'concat', 'average', 'set']
    models = []
    for m in model_types:
        model = Model(m, params, params_o, params_p, omics_df, protein_df, logger, 'test')
        models.append(model)

    # Split the samples for conditional generation and initialize training
    train_omics, test_omics = omics_data_splitter(
        omics_df, site, params.get('test_fraction', 0.2)
    )
    train_protein = protein_df
    test_protein = protein_df
    print("test omics:", len(test_omics), ":test protein", test_protein.shape, ":train omics:", len(train_omics), ":train protein:", train_protein.shape)

    logger.info('Models restored, start training.')

    # choose a validation cell line and protein.
    eval_cell_lines = [test_cell_line]
    eval_protein_names = np.random.choice(test_protein.index, size = 20, replace=False)
    eval_protein_names = [str(i) for i in eval_protein_names]

    for epoch in range(1, params['epochs'] + 1):
        for m in models:
            m.reset_metrics()
        
        for step in range(1, params['steps'] + 1):

            # Randomly sample a cell lines and proteins for training:
            cell_line  = [i for i in omics_df['cell_line'] if i != test_cell_line]
            protein_name = np.random.choice(train_protein.index, size = 20, replace=False)
            protein_name = [str(i) for i in protein_name]
            print(f'Current train cell_lines: {cell_line}')
            print(f'Current train proteins: {protein_name}')
            for m in models:
                m.train_and_save_steps(params['batch_size'], epoch, protein_name=protein_name, cell_line=cell_line)

        # Save model
        pt_name = [i for i in protein_name if i in unbiased_preds_df.columns]
        unbiased_predsP = unbiased_preds_df[pt_name].values.reshape(-1)[:params['batch_size']]
        unbiased_predsO = unbiased_preds_df[cell_line].values.reshape(-1)[:params['batch_size']]
        for m in models:
            m.model.save(f'gen_{epoch}.pt', f'enc_{epoch}_protein.pt', f'enc_{epoch}_omics.pt')
            m.save_loss_reward(epoch, params['epochs'])
            print(m.type)
            m.generate_and_save(epoch, 'train', params['batch_size'], unbiased_predsP=unbiased_predsP, protein_name=protein_name, 
                unbiased_predsO=unbiased_predsO, cell_line=cell_line)
        
        # Evaluate on a validation cell line and protein.
        pt_name = [i for i in eval_protein_names if i in unbiased_preds_df.columns]
        unbiased_predsP = unbiased_preds_df[pt_name].values.reshape(-1)[:params['eval_batch_size']]
        unbiased_predsO = unbiased_preds_df[eval_cell_lines].values.reshape(-1)[:params['eval_batch_size']]
        
        for m in models:
            print(m.type)
            m.generate_and_save(epoch, 'test', params['eval_batch_size'], unbiased_predsP=unbiased_predsP, protein_name=eval_protein_names, 
                unbiased_predsO=unbiased_predsO, cell_line=eval_cell_lines)

if __name__ == '__main__':
    main(parser_namespace=args)