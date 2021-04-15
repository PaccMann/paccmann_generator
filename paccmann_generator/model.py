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
#sys.path.append('/dataP/tol/paccmann_affinity')
sys.path.append('/home/tol/paccmann_affinity')
from paccmann_affinity.models.predictors import MODEL_FACTORY as MODEL_FACTORY_PROTEIN

from paccmann_generator.reinforce_proteins_omics import ReinforceProteinOmics
from paccmann_generator import ReinforceOmic
from paccmann_generator.reinforce_proteins import ReinforceProtein
from files import *

class Model:
    
    def __init__(self, modeltype, params, params_o, params_p, omics_df, protein_df, logger, model_folder_name=None):
        self.type = modeltype
        self.reset_metrics()
        self.model = self.create_combined_model(params, params_o, params_p, omics_df, protein_df, logger, model_folder_name)

    def reset_metrics(self):
        self.rewards, self.losses = [], []
        self.cell_steps, self.protein_steps, self.smiles_steps = [], [], []
        self.gen_mols ,self.gen_prot, self.gen_affinity = [], [], []
        self.gen_cell, self.gen_ic50, self.modes = [], [], []

    def create_combined_model(self, params, params_o, params_p, omics_df, protein_df, logger, model_folder_name):
        # Load languages
        generator_smiles_language = SMILESLanguage.load(
            os.path.join(mol_model_path, 'selfies_language.pkl')
        )

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

        #############################################
        # Create a fresh model that will be optimized
        # Restore SMILES Model
        with open(os.path.join(mol_model_path, 'model_params.json')) as f:
            mol_params = json.load(f)
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

        # Restore omics model
        with open(os.path.join(omics_model_path, 'model_params.json')) as f:
            cell_params = json.load(f)
        cell_encoder_rl = ENCODER_FACTORY['dense'](cell_params)
        cell_encoder_rl.load(
            os.path.join(
                omics_model_path,
                f"weights/best_{params.get('metric', 'both')}_encoder.pt"
            ),
            map_location=get_device()
        )
        cell_encoder_rl.eval()

        # Restore protein model
        with open(os.path.join(protein_model_path, 'model_params.json')) as f:
            protein_params = json.load(f)
        protein_encoder_rl = ENCODER_FACTORY['dense'](protein_params)
        protein_encoder_rl.load(
            os.path.join(
                protein_model_path,
                f"weights/best_{params.get('metric', 'both')}_encoder.pt"
            ),
            map_location=get_device()
        )
        protein_encoder_rl.eval()

        if (model_folder_name is None):
            model_folder_name = site + '_' + self.type + '_' + model_name + '_combined'
        if self.type =='set':
            adapted_keys = ['encoder.lstm.weights_x', 'encoder.lstm.weights_h', 'encoder.lstm.weights_c', 'encoder.lstm.bias', 'encoder.memory_mapping.weight', 'encoder.memory_mapping.bias', 'decoder.lstm.weights_x', 'decoder.lstm.weights_h', 'decoder.lstm.weights_c', 'decoder.lstm.bias', 'decoder.output_layer.weight', 'decoder.output_layer.bias', 'decoder.prob_layer.0.weight', 'decoder.prob_layer.0.bias']
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            #TO-DO: get the right encoder
            with open(os.path.join(set_encoder_path, 'train_params.json')) as f:
                encoder_params = json.load(f)
            #rename model keys
            new_state_dict = {}
            checkpoint = torch.load(set_encoder_path+'/setsae', map_location=device)
            state_dict = checkpoint["model_state_dict"]
            for i, (k,v) in enumerate(state_dict.items()):
                new_state_dict[adapted_keys[i]] = v
            setsae = SetsAE(device, **encoder_params).to(device)
            setsae.load_state_dict(new_state_dict)
            encoder_rl = setsae.encoder
            encoder_rl.latent_size = encoder_rl.hidden_size_encoder
            
            combined = ReinforceProteinOmics( 
                generator_rl, protein_encoder_rl, cell_encoder_rl, protein_predictor, \
                paccmann_predictor, protein_df, omics_df, params, params_o, params_p, generator_smiles_language, \
                model_folder_name, logger, remove_invalid, ensemble_type=self.type, set_encoder=encoder_rl)
        else:
            combined = ReinforceProteinOmics(
                generator_rl, protein_encoder_rl, cell_encoder_rl, \
                protein_predictor, paccmann_predictor, protein_df, omics_df, params, params_o, params_p, \
                generator_smiles_language, model_folder_name, logger, remove_invalid, ensemble_type=self.type
            )
        return combined
          
    def save_loss_reward(self, epoch, tot_epochs):
        #Plot loss development
        if(self.protein_steps is None):
            loss_df = pd.DataFrame({'loss': self.losses, 'rewards': self.rewards, 'smiles':self.smiles_steps, 'cell_line':self.cell_steps, 'epoch':epoch})
        elif(self.cell_steps is None):
            loss_df = pd.DataFrame({'loss': self.losses, 'rewards': self.rewards, 'smiles':self.smiles_steps, 'proteins':self.protein_steps, 'epoch':epoch})
        else:
            loss_df = pd.DataFrame({'loss': self.losses, 'rewards': self.rewards, 'smiles':self.smiles_steps, 'proteins':self.protein_steps, 'cell_line':self.cell_steps,  'epoch':epoch})
        if epoch ==1:
            loss_df.to_csv(self.model.model_path + '/results/loss_reward_evolution.csv')
        else:
            loss_df.to_csv(self.model.model_path + '/results/loss_reward_evolution.csv', mode='a', header=False)
        losses_rewards_all = pd.read_csv(self.model.model_path + '/results/loss_reward_evolution.csv', header = 0)
        rewards_all = losses_rewards_all['rewards']
        losses_all = losses_rewards_all['loss']
        plot_loss(
            losses_all,
            rewards_all,
            tot_epochs,
            #protein_name,
            self.model.model_path,
            rolling=5
        )
        
    def train_and_save_steps(self, batch_size, epoch, protein_name=None, cell_line=None):
        if (protein_name is None):
            rew, loss, smiles_step, smiles_idx_steps = self.model.policy_gradient(
                cell_line, epoch, batch_size
            )
            cell = (cell_line*batch_size)[:batch_size]
            cell = [cell[idx] for idx in smiles_idx_steps]
            self.cell_steps.append(cell)
        elif (cell_line is None):
            rew, loss, smiles_step, smiles_idx_steps = self.model.policy_gradient(
                protein_name, epoch, batch_size
            )
            proteins = (protein_name*batch_size)[:batch_size]
            proteins = [proteins[idx] for idx in smiles_idx_steps]
            self.protein_steps.append(proteins)
        else:
            #### TO-DO change policy_gradient function of combined model to be similar to single ones.
            rew, loss, smiles_step, smiles_idx_steps = self.model.policy_gradient(
                protein_name, cell_line, epoch, batch_size
            )
            proteins = (protein_name*batch_size)[:batch_size]
            proteins = [proteins[idx] for idx in smiles_idx_steps]
            self.protein_steps.append(proteins)
            cell = (cell_line*batch_size)[:batch_size]
            cell = [cell[idx] for idx in smiles_idx_steps]
            self.cell_steps.append(cell)

        print(f"Epoch {epoch:d}")
        self.rewards.append(rew.item())
        self.losses.append(loss)
        self.smiles_steps.append(smiles_step)

    def generate_and_save(self, epoch, mode, param, unbiased_predsP=None, unbiased_predsO=None, protein_name=None, cell_line=None):
        dict_res = {}
        if (cell_line is None):
            smiles, predsP, idx = self.model.generate_compounds_and_evaluate(
                epoch, param, protein_name
            )
        elif(protein_name is None):
            smiles, predsO, idx = self.model.generate_compounds_and_evaluate(
                epoch, param, cell_line
            )
        else:
            smiles, predsP, predsO, idx = self.model.generate_compounds_and_evaluate(
                epoch, param, protein_name, cell_line
            )
        
        if(cell_line is not None):
            cells = (cell_line*param)[:param]
            cells = [cells[i] for i in idx]
            cells = [o for i,o in enumerate(cells) if (predsO[i] < self.model.ic50_threshold)]
        if(protein_name is not None):
            proteins = (protein_name*param)[:param]
            proteins = [proteins[i] for i in idx]
            proteins = [p for i, p in enumerate(proteins) if predsP[i] > 0.5]

        if (cell_line is None):
            gs = [s for i, s in enumerate(smiles) if predsP[i] > 0.5]
            gp_p = preds[predsP > 0.5]
            for p_p, p in zip(gp_p, proteins):
                self.gen_prot.append(p)
                self.gen_affinity.append(p_p)
            dict_res = {
                'protein': self.gen_prot,
                'Binding probability': self.gen_affinity
            }
            plot_and_compare_proteins(
                unbiased_predsP, predsP, protein_name, epoch, self.model.model_path,
                mode, param
            )
        elif(protein_name is None):
            gs = [
                s for i, s in enumerate(smiles)
                if predsO[i] < self.model.ic50_threshold
            ]
            gp_o = predsO[predsO < self.model.ic50_threshold]
            for p_o, c in zip(gp_o, cells):
                self.gen_cell.append(c)
                self.gen_ic50.append(p_o)
            dict_res = {
                'cell_line': self.gen_cell,
                'IC50': self.gen_ic50
            }
            plot_and_compare(
                unbiased_predsO, predsO, site, cell_line, epoch, self.model.model_path,
                mode, param
            )
        else:
            gs = [
                s for i, s in enumerate(smiles)
                if predsO[i] < self.model.ic50_threshold and predsP[i] > 0.5
            ]
            gp_o = predsO[(predsO < self.model.ic50_threshold) & (predsP > 0.5)]
            gp_p = predsP[(predsO < self.model.ic50_threshold) & (predsP > 0.5)]
            for p_o, p_p, p, c in zip(gp_o, gp_p, proteins, cells):
                self.gen_cell.append(c)
                self.gen_prot.append(p)
                self.gen_affinity.append(p_p)
                self.gen_ic50.append(p_o)
            dict_res = {
                'protein': self.gen_prot,
                'Binding probability': self.gen_affinity,
                'cell_line': self.gen_cell,
                'IC50': self.gen_ic50
            }
            plot_and_compare(
                unbiased_predsO, predsO, site, cell_line, epoch, self.model.model_path,
                mode, param
            )
            plot_and_compare_proteins(
                unbiased_predsP, predsP, protein_name, epoch, self.model.model_path,
                mode, param
            )

        for s in gs:
            self.gen_mols.append(s)
            self.modes.append(mode)
        dict_res['SMILES'] = self.gen_mols
        dict_res['mode'] = self.modes
        dict_res['epoch'] = [epoch] * len(self.gen_mols)

        
        if mode == 'test':
            df = pd.DataFrame(dict_res)
            if epoch ==1:
                df.to_csv(os.path.join(self.model.model_path, 'results', 'generated.csv'))
            else:
                df.to_csv(os.path.join(self.model.model_path, 'results', 'generated.csv'), mode='a', header=False)
        