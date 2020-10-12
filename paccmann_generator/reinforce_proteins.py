"""PaccMann^RL: Protein-driven drug generation"""
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from .drug_evaluators import (
    QED, SCScore, ESOL, SAS, Lipinski, Tox21, SIDER, ClinTox, OrganDB
)
from rdkit import Chem
from pytoda.transforms import LeftPadding, ToTensor

from .reinforce import Reinforce


class ReinforceProtein(Reinforce):

    def __init__(
        self, generator, encoder, predictor, protein_df, params, generator_smiles_language, model_name,
        logger
    ):
        """
        Constructor for the Reinforcement object.

        Args:
            generator (nn.Module): SMILES generator object.
            encoder (nn.Module): A protein encoder.
            predictor (nn.Module): A binding affinity predictor
            protein_df (pd.Dataframe): Protein sequences of interest.
            params: dict with hyperparameter.
            model_name: name of the model.
            logger: a logger.

        Returns:
            object of type Reinforcement used for biasing the properties
            estimated by the predictor of trajectories produced by the
            generator to maximize the custom reward function get_reward.
        """

        super(ReinforceProtein, self).__init__(
            generator, encoder, params, model_name, logger
        )  # yapf: disable

        self.affinity_predictor = predictor
        self.affinity_predictor.eval()

        self.protein_df = protein_df

        self.pad_smiles_predictor = LeftPadding(
            predictor.smiles_padding_length,
            predictor.smiles_language.padding_index
        )
        self.pad_protein_predictor = LeftPadding(
            predictor.protein_padding_length,
            predictor.protein_language.padding_index
        )

        self.generator_smiles_language = generator_smiles_language

        self.protein_to_tensor = ToTensor(self.device)
        self.update_params(params)

        self.tox21 = Tox21(
            params.get(
                'tox21_path',
                os.path.join(
                    os.path.expanduser('~'), 'Box', 'Molecular_SysBio', 'data',
                    'cytotoxicity', 'models', 'Tox21_deepchem'
                )
            )
        )

    def update_params(self, params):
        """Update parameter

        Args:
            params (dict): Dict with (new) parameters
        """
        # parameter for reward function
        self.qed = QED()
        self.scscore = SCScore()
        self.esol = ESOL()
        self.sas = SAS()
        self.lipinski = Lipinski()
        self.clintox = ClinTox(
            params.get(
                'clintox_path',
                os.path.join(
                    os.path.expanduser('~'), 'Box', 'Molecular_SysBio', 'data',
                    'cytotoxicity', 'models', 'ClinToxMulti'
                )
            )
        )
        self.sider = SIDER(
            params.get(
                'sider_path',
                os.path.join(
                    os.path.expanduser('~'), 'Box', 'Molecular_SysBio', 'data',
                    'cytotoxicity', 'models', 'Sider'
                )
            )
        )
        self.tox21 = Tox21(
            params.get(
                'tox21_path',
                os.path.join(
                    os.path.expanduser('~'), 'Box', 'Molecular_SysBio', 'data',
                    'cytotoxicity', 'models', 'Tox21_deepchem'
                )
            )
        )
        self.organdb = OrganDB(
            params.get(
                'organdb_path',
                os.path.join(
                    os.path.expanduser('~'), 'Box', 'Molecular_SysBio', 'data',
                    'cytotoxicity', 'models', 'Organdb_github'
                )
            ), params['site']
        )

        self.update_reward_fn(params)
        # discount factor
        self.gamma = params.get('gamma', 0.99)
        # maximal length of generated molecules
        self.generate_len = params.get(
            'generate_len', self.affinity_predictor.params['smiles_padding_length'] - 2
        )
        # smoothing factor for softmax during token sampling in decoder
        self.temperature = params.get('temperature', 0.8)
        # gradient clipping in decoder
        self.grad_clipping = params.get('clip_grad', None)

    def encode_protein(self, protein=None, batch_size=128):
        """
        Encodes protein in latent space with protein encoder.
        Args:
            protein (str): Name of a protein
            batch_size (int): batch_size
        """
        if protein is None:
            latent_z = torch.randn(1, batch_size, self.encoder.latent_size)
        else:
            protein_mu = []
            protein_logvar = []
            for prot in protein:
                protein_encoder_tensor, _ = (
                    self.protein_to_numerical(
                        prot, encoder_uses_sequence=False, predictor_uses_sequence=True
                    )
                )
                protein_mu_i, protein_logvar_i = self.encoder(protein_encoder_tensor)
                protein_logvar.append(torch.unsqueeze(protein_logvar_i, 0).detach().numpy()[0][0])
                protein_mu.append(torch.unsqueeze(protein_mu_i, 0).detach().numpy()[0][0])
            protein_mu = torch.as_tensor(protein_mu)
            protein_logvar = torch.as_tensor(protein_logvar)
            
            protein_mu_batch = protein_mu.repeat(batch_size, 1)
            protein_logvar = protein_logvar.repeat(batch_size, 1)
            if protein_mu_batch.size()[0]>batch_size:
                protein_mu_batch = protein_mu_batch[:batch_size]
                protein_logvar = protein_logvar[:batch_size]

            latent_z = torch.unsqueeze(
                self.reparameterize(
                    protein_mu_batch,
                    protein_logvar
                ), 0
            )
        return latent_z

    def protein_to_numerical(
        self,
        protein,
        encoder_uses_sequence=True,
        predictor_uses_sequence=True
    ):
        """
        Receives a name of a protein.
        Returns two numerical torch Tensor, the first for the protein encoder,
        the second for the affinity predictor.
        Args:
            protein (str): Name of the protein
            encoder_uses_sequence (bool): Whether the encoder uses the protein
                sequence or an embedding.
            predictor_uses_sequence (bool): Whether the predictor uses the
                protein sequence or an embedding.

        """
        protein_sequence = self.protein_df.loc[protein]['sequence']
        if predictor_uses_sequence:
            sequence_tensor_p = torch.unsqueeze(
                self.protein_to_tensor(
                    self.pad_protein_predictor(
                        self.affinity_predictor.protein_language.
                        sequence_to_token_indexes(protein_sequence) 
                    )
                ), 0
            )
        if encoder_uses_sequence:
            sequence_tensor_e = torch.unsqueeze(
                self.protein_to_tensor(
                    self.pad_protein_predictor(
                        self.encoder.protein_language.
                        sequence_to_token_indexes(protein_sequence)
                    )
                ), 0
            )
        if (not encoder_uses_sequence) or (not predictor_uses_sequence):
            # Column names of DF
            locations = [str(x) for x in range(768)]
            protein_encoding = self.protein_df.loc[protein][locations]
            encoding_tensor = torch.unsqueeze(
                torch.Tensor(protein_encoding), 0
            )
        t1 = sequence_tensor_e if encoder_uses_sequence else encoding_tensor
        t2 = sequence_tensor_p if predictor_uses_sequence else encoding_tensor
        return t1, t2

    def generate_compounds_and_evaluate(
        self,
        epoch,
        batch_size,
        protein=None,
        primed_drug=' ',
        return_latent=False,
        remove_invalid=True
    ):
        """
        Generate some compounds and evaluate them with the predictor

        Args:
            epoch (int): The training epoch.
            batch_size (int): The batch size.
            protein (str): A string, the protein used to drive generator.
            primed_drug (str): SMILES string to prime the generator.

        Returns:
            np.array: Predictions from PaccMann.
        """
        if primed_drug != ' ':
            raise ValueError('Drug priming not yet supported.')

        self.affinity_predictor.eval()
        self.encoder.eval()
        self.generator.eval()

        if protein is None:
            # Generate a random molecule
            latent_z = torch.randn(
                1, batch_size, self.generator.decoder.latent_dim
            )
        else:
            protein_mu = []
            protein_logvar = []
            protein_predictor_tensor = []
            for prot in protein:
                protein_encoder_tensor, protein_predictor_tensor_i = (
                    self.protein_to_numerical(
                        prot, encoder_uses_sequence=False, predictor_uses_sequence=True
                    )
                )
                protein_mu_i, protein_logvar_i = self.encoder(protein_encoder_tensor)
                protein_predictor_tensor.append(torch.unsqueeze(protein_predictor_tensor_i, 0).detach().numpy()[0][0])
                protein_logvar.append(torch.unsqueeze(protein_logvar_i, 0).detach().numpy()[0][0])
                protein_mu.append(torch.unsqueeze(protein_mu_i, 0).detach().numpy()[0][0])
            protein_mu = torch.as_tensor(protein_mu)
            protein_logvar = torch.as_tensor(protein_logvar)
            protein_predictor_tensor = torch.as_tensor(protein_predictor_tensor)
            
            protein_mu_batch = protein_mu.repeat(batch_size, 1)
            protein_logvar = protein_logvar.repeat(batch_size, 1)
            protein_predictor_tensor = protein_predictor_tensor.repeat(batch_size, 1)
            if protein_mu_batch.size()[0]>batch_size:
                protein_mu_batch = protein_mu_batch[:batch_size]
                protein_logvar = protein_logvar[:batch_size]
                protein_predictor_tensor = protein_predictor_tensor[:batch_size]
            
            latent_z = torch.unsqueeze(
                self.reparameterize(
                    protein_mu_batch,
                    protein_logvar
                ), 0
            )

        # Generate drugs
        valid_smiles, valid_nums, valid_idx = self.get_smiles_from_latent(
            latent_z, remove_invalid=remove_invalid
        )

        smiles_t = self.smiles_to_numerical(valid_smiles, target='predictor')

        # Evaluate drugs
        pred, pred_dict = self.affinity_predictor(
            smiles_t, protein_predictor_tensor[valid_idx]
        )
        pred = np.squeeze(pred.detach().numpy())
        #self.plot_hist(log_preds, cell_line, epoch, batch_size)

        if return_latent:
            return valid_smiles, pred, latent_z
        else:
            return valid_smiles, pred, valid_idx

    def update_reward_fn(self, params):
        """ Set the reward function
        
        Arguments:
            params {dict} -- Hyperparameter for PaccMann reward function


        """
        super().update_reward_fn(params)
        self.affinity_weight = params.get('affinity_weight', 1.)
        self.tox21_weight = params.get('tox21_weight', .5)

        def tox_f(s):
            x = 0
            if self.tox21_weight > 0.:
                x += self.tox21_weight * self.tox21(s)
            if self.sider_weight > 0.:
                x += self.sider_weight * self.sider(s)
            if self.clintox_weight > 0.:
                x += self.clintox_weight * self.clintox(s)
            if self.organdb_weight > 0.:
                x += self.organdb_weight * self.organdb(s)
            return x

        # This is the joint reward function. Each score is normalized to be
        # inside the range [0, 1].
        self.reward_fn = (
            lambda smiles, protein, valid_idx, batch_size: (
                self.affinity_weight * self.
                get_reward_affinity(smiles, protein, valid_idx, batch_size) + np.
                array([tox_f(s) for s in smiles])
            )
        )

    def get_reward(self, valid_smiles, protein, valid_idx, batch_size):
        """Get the reward
        
        Arguments:
            valid_smiles (list): A list of valid SMILES strings.
            protein (str): Name of the target protein
        
        Returns:
            np.array: computed reward.
        """
        return self.reward_fn(valid_smiles, protein, valid_idx, batch_size)

    def get_reward_affinity(self, valid_smiles, protein, idx, batch_size):
        """
        Get the reward from affinity predictor

        Args:
            valid_smiles (list): A list of valid SMILES strings.
            protein (str): Name of target protein

        Returns:
            np.array: computed reward (fixed to 1/(1+exp(x))).
        """
        # Build up SMILES tensor and GEP tensor
        smiles_tensor = self.smiles_to_numerical(
            valid_smiles, target='predictor'
        )

        # If all SMILES are invalid, no reward is given
        if len(smiles_tensor) == 0:
            return 0

        protein_tensor = []
        for prot in protein:
            _, protein_tensor_i = self.protein_to_numerical(
                prot, encoder_uses_sequence=False
            )
            protein_tensor.append(torch.unsqueeze(protein_tensor_i, 0).detach().numpy()[0][0])
        protein_tensor = torch.as_tensor(protein_tensor)
        protein_tensor = protein_tensor.repeat(batch_size, 1)
        if protein_tensor.size()[0]>batch_size:
            protein_tensor = protein_tensor[:batch_size]
        pred, pred_dict = self.affinity_predictor(
            smiles_tensor, protein_tensor[idx]
        )

        return np.squeeze(pred.detach().numpy())

    def smiles_to_numerical(self, smiles_list, target='predictor'):
        """
        Receives a list of SMILES.
        Converts it to a numerical torch Tensor according to smiles_language
        """

        if target == 'generator':
            # NOTE: Code for this in the normal REINFORCE class
            raise ValueError('Priming drugs not yet supported')

        # TODO: Workaround since predictor does not understand aromatic carbons
        smiles_list = [
            Chem.MolToSmiles(
                Chem.MolFromSmiles(s, sanitize=False), kekuleSmiles=True
            ).replace(':', '') for s in smiles_list
        ]

        # Convert strings to numbers and padd length.
        smiles_num = [
            torch.unsqueeze(
                self.smiles_to_tensor(
                    self.pad_smiles_predictor(
                        self.affinity_predictor.smiles_language.
                        smiles_to_token_indexes(smiles)
                    )
                ), 0
            ) for smiles in smiles_list
        ]

        # Catch scenario where all SMILES are invalid
        if len(smiles_num) == 0:
            return torch.Tensor()

        smiles_tensor = torch.cat(smiles_num, dim=0)
        return smiles_tensor

    def policy_gradient(self, protein, epoch, batch_size=10):
        """
        Implementation of the policy gradient algorithm.

        Args:
            protein (str): Name of protein to generate drugs.
            epoch (int): training epoch.
            batch_size (int): batch size.

        Returns:
            tuple(float, float): total reward and total loss.
        """
        rl_loss = 0
        self.optimizer.zero_grad()

        # Encode the protein
        latent_z = self.encode_protein(protein, batch_size)

        # Produce molecules
        valid_smiles, valid_nums, valid_idx = self.get_smiles_from_latent(
            latent_z, remove_invalid=True
        )

        # Get rewards (list, one reward for each valid smiles)
        rewards = self.get_reward(valid_smiles, protein, valid_idx, batch_size)
        reward_tensor = torch.unsqueeze(torch.Tensor(rewards), 1)

        # valid_nums is a list of torch.Tensor, each with varying length,
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

        # # Compute loss
        output = torch.Tensor()
        for p in range(len(padded_nums) - 1):
            output, hidden, stack = self.generator.decoder(
                torch.unsqueeze(padded_nums[p], 0), hidden, stack
            )
            output = self.generator.decoder.output_layer(output).squeeze()

            log_probs = F.log_softmax(output, dim=1)
            target_char = torch.unsqueeze(padded_nums[p + 1], 1)
            rl_loss -= torch.mean(
                log_probs.gather(1, target_char) * reward_tensor
            )
            # discounted_rewards = discounted_rewards * self.gamma
        summed_reward = torch.mean(torch.Tensor(rewards))
        rl_loss.backward()

        if self.grad_clipping is not None:
            torch.nn.utils.clip_grad_norm_(
                list(self.generator.decoder.parameters()) +
                list(self.encoder.parameters()), self.grad_clipping
            )
        self.optimizer.step()
        return summed_reward, rl_loss.item()

