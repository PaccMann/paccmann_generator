"""PaccMann^RL: Multi-modal drug generation"""
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from paccmann_generator.drug_evaluators import QED, Tox21
from paccmann_generator.reinforce import Reinforce
from pytoda.transforms import LeftPadding, ToTensor


class ReinforceMultiModalSets(Reinforce):

    def __init__(
        self, generator: nn.Module, encoder: nn.Module, affinity_predictor: nn.Module,
        efficacy_predictor: nn.Module, protein_df: pd.DataFrame, gep_df: pd.DataFrame,
        params: dict, model_name: str, logger
    ):
        """
        Constructor for the Reinforcement object.

        Args:
            generator (nn.Module): SMILES generator object.
            encoder (nn.Module): The encoder component of the set autoencoder.
            affinity_predictor (nn.Module): A binding affinity predictor
            efficacy_predictor (nn.Module): A IC50 predictor, i.e. PaccMann (MCA object).
            protein_df (pd.Dataframe): A Pandas df with the protein sequences of
                interest and their latent embeddings.
            gep_df (pd.DataFrame): A pandas df with gene expression profiles and their
                latent embeddings of cancer cells.
                GEP values need to be ordered, s.t. both PaccMann and encoder
                understand it.
            params (dict): Hyperparameter dictionary.
            model_name (str): name of the model.
            logger: a logger.

        Returns:
            object of type Reinforcement used for biasing the properties
            estimated by the predictor of trajectories produced by the
            generator to maximize the custom reward function get_reward.
        """

        super(ReinforceMultiModalSets, self).__init__(
            generator, encoder, params, model_name, logger
        )  # yapf: disable

        self.affinity_predictor = affinity_predictor
        self.affinity_predictor.eval()

        self.efficacy_predictor = efficacy_predictor
        self.efficacy_predictor.eval()

        self.protein_df = protein_df
        self.gep_df = gep_df

        self.pad_efficacy_smiles_predictor = LeftPadding(
            params['predictor_smiles_length'],
            efficacy_predictor.smiles_language.padding_index
        )
        self.pad_affinity_smiles_predictor = LeftPadding(
            affinity_predictor.smiles_padding_length,
            affinity_predictor.smiles_language.padding_index
        )

        self.pad_protein_predictor = LeftPadding(
            affinity_predictor.protein_padding_length,
            affinity_predictor.protein_language.padding_index
        )

        self.protein_to_tensor = ToTensor(self.device)

        self.update_params(params)

        self.generate_len = params.get(
            'generate_len', self.efficacy_predictor.params['smiles_padding_length'] - 2
        )

        self.temperature = params.get('temperature', 1.0)
        # gradient clipping in decoder
        self.grad_clipping = params.get('clip_grad', None)

    def update_params(self, params: dict):
        """Update parameter

        Args:
            params (dict): Dict with (new) parameters
        """

        self.ic50_min = params.get('IC50_min', -8.77435)
        self.ic50_max = params.get('IC50_max', 11.83146)

        self.ic50_threshold = params.get('IC50_threshold', 2.0)

        self.rewards = params.get('rewards', (11, 1))

        self.update_reward_fn(params)

        self.qed = QED()

        if self.tox21_weight > 0.:
            self.tox21 = Tox21(
                params.get('tox21_path', os.path.join('..', 'data', 'models', 'Tox21'))
            )

        self.gamma = params.get('gamma', 0.99)

    def get_log_molar(self, y):
        """
        Converts PaccMann predictions from [0,1] to log(micromolar) range.
        """
        return y * (self.ic50_max - self.ic50_min) + self.ic50_min

    def get_set(
        self, cell_line: str, protein: str, batch_size: int, shuffle=True
    ) -> Tuple:
        """Combines the cell line and protein into a set.

        Args:
            cell_line (str): Name of cell line.
            protein (str): Name of protein.
            batch_size (int): Batch size.
            shuffle (bool, optional): Whether to shuffle the set. Defaults to True.

        Returns:
            Tuple: Tuple containing the set, the latent gene expression profile of the
                cell line and protein sequence to pass into the affinity predictor.
        """

        if protein is None:
            protein_encoder_tensor = torch.randn(1, self.encoder.latent_size)
            protein_predictor_tensor = None
        else:
            protein_encoder_tensor, protein_predictor_tensor = (
                self.protein_to_numerical(
                    protein, encoder_uses_sequence=False, predictor_uses_sequence=True
                )
            )

        locations = ['latent_' + str(x) for x in range(128)]

        if cell_line is None:
            gep_t = torch.randn(1, self.encoder.latent_size)
        else:
            gep_t = torch.Tensor(
                self.gep_df.loc[self.gep_df['cell_line'] == cell_line][locations].values
            )

        combined_set = torch.stack(
            [gep_t.repeat(batch_size, 1),
             protein_encoder_tensor.repeat(batch_size, 1)]
        ).permute(1, 0, 2)
        if shuffle:
            combined_set = combined_set[:, torch.randperm(2), :]

        return combined_set, gep_t, protein_predictor_tensor

    def protein_to_numerical(
        self,
        protein: str,
        encoder_uses_sequence: bool = True,
        predictor_uses_sequence: bool = True
    ) -> Tuple:
        """
        Receives a name of a protein.
        Returns two numerical torch Tensor, the first for the protein encoder,
        the second for the affinity predictor.
        Args:
            protein (str): Name of the protein.
            encoder_uses_sequence (bool, optional): Whether the encoder uses the protein
                sequence or an embedding. Defaults to True.
            predictor_uses_sequence (bool, optional): Whether the predictor uses the
                protein sequence or an embedding. Defaults to True.

        Returns:
            Tuple: Tuple containing the protein sequences for the encoder and predictor.
        """

        if encoder_uses_sequence or predictor_uses_sequence:
            protein_sequence = self.protein_df.loc[protein]['Sequence']
            # TODO: This may cause bugs if encoder_uses_sequence is True and
            # uses another protein language object
            sequence_tensor = torch.unsqueeze(
                self.protein_to_tensor(
                    self.pad_protein_predictor(
                        self.affinity_predictor.protein_language.
                        sequence_to_token_indexes(protein_sequence)
                    )
                ), 0
            )
        if (not encoder_uses_sequence) or (not predictor_uses_sequence):
            # Column names of DF
            locations = [str(x) for x in range(128)]
            protein_encoding = self.protein_df.loc[protein][locations]
            encoding_tensor = torch.unsqueeze(torch.Tensor(protein_encoding), 0)
        t1 = sequence_tensor if encoder_uses_sequence else encoding_tensor
        t2 = sequence_tensor if predictor_uses_sequence else encoding_tensor
        return t1, t2

    def generate_compounds_and_evaluate(
        self,
        epoch: int,
        batch_size: int,
        protein: str = None,
        cell_line: str = None,
        primed_drug: str = ' ',
        return_latent: bool = False
    ) -> Tuple:
        """
        Generate some compounds and evaluate them with the predictor

        Args:
            epoch (int): The training epoch.
            batch_size (int): The batch size.
            protein (str): A string, the protein used to drive generator.
            cell_line (str): Name of cell line.
            primed_drug (str): SMILES string to prime the generator.
            return_latent (bool, optional): whether to return the latent code or the
                valid smiles indices. Defaults to False.

        Returns:
            Tuple: Tuple containing predictions from PaccMann and Binding Affinity
                predictor, as well as the valid smiles and either its indices or the
                latent code.
        """
        if primed_drug != ' ':
            raise ValueError('Drug priming not yet supported.')

        self.affinity_predictor.eval()
        self.efficacy_predictor.eval()

        self.encoder.eval()
        self.generator.eval()

        input_set, gene_embedding, protein_predictor = self.get_set(
            cell_line, protein, batch_size, shuffle=True
        )
        latent_z, hn, rn = self.encoder(input_set)

        # Generate drugs
        valid_smiles, valid_nums, valid_idx = self.get_smiles_from_latent(latent_z)

        smiles_t_affinity = self.smiles_to_numerical(valid_smiles, target='affinity')
        smiles_t_efficacy = self.smiles_to_numerical(valid_smiles, target='efficacy')

        locations = ['gene_expression_' + str(x) for x in range(2128)]

        gep = torch.Tensor(
            self.gep_df.loc[self.gep_df['cell_line'] == cell_line  # yapf: disable
                            ][locations].values
        )

        pred_p, pred_dict_p = self.affinity_predictor(
            smiles_t_affinity, protein_predictor.repeat(smiles_t_affinity.shape[0], 1)
        )
        pred_p = np.squeeze(pred_p.detach().numpy())

        pred_o, pred_dict_o = self.efficacy_predictor(
            smiles_t_efficacy, gep.repeat(smiles_t_efficacy.shape[0], 1)
        )
        log_preds = self.get_log_molar(np.squeeze(pred_o.detach().numpy()))

        if return_latent:
            return valid_smiles, pred_p, log_preds, latent_z
        else:
            return valid_smiles, pred_p, log_preds, valid_idx

    def update_reward_fn(self, params: dict):
        """ Set the reward function
        Arguments:
            params (dict): Hyperparameter for PaccMann reward function
        """

        self.qed_weight = params.get('qed_weight', 0.)

        self.tox21_weight = params.get('tox21_weight', .5)

        self.paccmann_weight = params.get('paccmann_weight', 1.)
        self.affinity_weight = params.get('affinity_weight', 1.)

        self.weight_tot = (
            self.paccmann_weight + self.affinity_weight + self.tox21_weight +
            self.qed_weight
        )

        self.reward_fn = (
            lambda smiles, protein, cell: (
                self.affinity_weight / self.weight_tot * self.
                get_reward_affinity(smiles, protein) + self.paccmann_weight / self.
                weight_tot * self.get_reward_paccmann(smiles, cell) + np.array(
                    [
                        self.qed_weight / self.weight_tot * self.qed(s) + self.
                        tox21_weight / self.weight_tot * self.tox21(s) for s in smiles
                    ]
                )
            )
        )

    def get_reward(self, valid_smiles: List, protein: str, cell_line: str) -> np.array:
        """Get the reward

        Arguments:
            valid_smiles (list): A list of valid SMILES strings.
            protein (str): Name of the target protein.
            cell_line (str): Name of cell line.

        Returns:
            np.array: Computed reward.
        """
        return self.reward_fn(valid_smiles, protein, cell_line)

    def smiles_to_numerical(self, smiles_list: List, target: str) -> torch.Tensor:
        """Receives a list of SMILES and converts it to a numerical torch Tensor
            according to smiles_language.
        
        Args:
            smiles_list (List): List of SMILES.
            target (str): One of efficacy or affinity to determine which predictor to
                use.
        
        Returns:
            torch.Tensor: Tensor of SMILES.
        """

        if target == 'generator':
            raise ValueError('Priming drugs not yet supported')

        if target == 'efficacy':
            smiles_num = [
                torch.unsqueeze(
                    self.smiles_to_tensor(
                        self.pad_efficacy_smiles_predictor(
                            self.efficacy_predictor.smiles_language.
                            smiles_to_token_indexes(smiles)
                        )
                    ), 0
                ) for smiles in smiles_list
            ]
        elif target == 'affinity':
            smiles_num = [
                torch.unsqueeze(
                    self.smiles_to_tensor(
                        self.pad_affinity_smiles_predictor(
                            self.affinity_predictor.smiles_language.
                            smiles_to_token_indexes(smiles)
                        )
                    ), 0
                ) for smiles in smiles_list
            ]

        if len(smiles_num) == 0:
            return torch.Tensor()

        smiles_tensor = torch.cat(smiles_num, dim=0)
        if target == 'efficacy':
            smiles_tensor = smiles_tensor.narrow(1, len(smiles_tensor[0]) - 250, 250)

        return smiles_tensor

    def get_reward_affinity(self, valid_smiles: List, protein: str) -> np.array:
        """Get the reward from affinity predictor.
        
        Args:
            valid_smiles (List): A list of valid SMILES strings.
            protein (str): Name of target protein

        Returns:
            np.array: computed reward (fixed to 1/(1+exp(x))).
        """

        smiles_tensor = self.smiles_to_numerical(valid_smiles, target='affinity')

        if len(smiles_tensor) == 0:
            return 0

        _, protein_tensor = self.protein_to_numerical(
            protein, encoder_uses_sequence=False
        )

        pred, pred_dict = self.affinity_predictor(
            smiles_tensor, protein_tensor.repeat(smiles_tensor.shape[0], 1)
        )

        return np.squeeze(pred.detach().numpy())

    def get_reward_paccmann(self, valid_smiles: List, cell_line: str) -> np.array:
        """Get the reward from PaccMann.
        
        Args:
            valid_smiles (List): A list of valid SMILES strings.
            cell_line (str): Name of the cell line.
        Returns:
            np.array: computed reward (fixed to 1/(1+exp(x))).
        """

        smiles_tensor = self.smiles_to_numerical(valid_smiles, target='efficacy')

        if len(smiles_tensor) == 0:
            return 0

        locations = ['gene_expression_' + str(x) for x in range(2128)]

        gep_t = torch.Tensor(
            self.gep_df.loc[self.gep_df['cell_line'] == cell_line  # yapf: disable
                            ][locations].values
        )

        pred, pred_dict = self.efficacy_predictor(
            smiles_tensor, gep_t.repeat(len(valid_smiles), 1)
        )
        log_micromolar_pred = self.get_log_molar(np.squeeze(pred.detach().numpy()))
        lmps = [lmp if lmp < self.ic50_threshold else 10 for lmp in log_micromolar_pred]
        return 1 / (1 + np.exp(lmps))

    def policy_gradient(
        self, cell_line: str, protein: str, epoch: int, batch_size: int = 10
    ) -> Tuple:
        """Implementation of the policy gradient algorithm.
        
        Args:
            protein (str): Name of protein to generate drugs.
            cell_line (str): Name of cell line to generate drugs.
            epoch (int): training epoch.
            batch_size (int, optional): Batch size. Defaults to 10.

        Returns:
            Tuple: Tuple containing the total reward and total loss.
        """
        rl_loss = 0
        self.optimizer.zero_grad()

        # get set
        input_set, gep_t, protein_predictor = self.get_set(
            cell_line, protein, batch_size, shuffle=True
        )
        # Encode the set
        encoder_output = self.encoder(input_set)
        # The encoder output is a tuple of the cell's internal states and the read vector
        # The latent_z can be any combination of these tuple elements. The mean of these
        # tensors is used as the latent_z here.
        latent_z = torch.mean(torch.stack(encoder_output), dim=0)
        # Produce molecules
        valid_smiles, valid_nums, valid_idx = self.get_smiles_from_latent(latent_z)

        # Get rewards (list, one reward for each valid smiles)
        rewards = self.get_reward(valid_smiles, protein, cell_line)
        reward_tensor = torch.unsqueeze(torch.Tensor(rewards), 1)

        padded_nums = torch.nn.utils.rnn.pad_sequence(valid_nums)
        num_mols = padded_nums.shape[1]

        self.generator.decoder._update_batch_size(num_mols)

        # Batch processing
        lrps = 1
        if self.generator.decoder.latent_dim == 2 * self.encoder.latent_size:
            lrps = 2
        hidden = self.generator.decoder.latent_to_hidden(
            latent_z.repeat(
                self.generator.decoder.n_layers, 1, lrps
            )[:, valid_idx, :]
        )  # yapf: disable
        stack = self.generator.decoder.init_stack

        # Compute loss
        output = torch.Tensor()
        for p in range(len(padded_nums) - 1):
            output, hidden, stack = self.generator.decoder(
                torch.unsqueeze(padded_nums[p], 0), hidden, stack
            )
            output = self.generator.decoder.output_layer(output).squeeze()

            log_probs = F.log_softmax(output, dim=1)
            target_char = torch.unsqueeze(padded_nums[p + 1], 1)
            rl_loss -= torch.mean(log_probs.gather(1, target_char) * reward_tensor)

        summed_reward = torch.mean(torch.Tensor(rewards))
        rl_loss.backward()

        self.optimizer.step()
        return summed_reward, rl_loss.item()
