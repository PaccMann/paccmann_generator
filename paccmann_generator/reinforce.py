"""PaccMann^RL: Policy gradient class (implemented through REINFORCE)."""
import json
import os
import warnings
from math import floor, log
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
import matplotlib.pyplot as plt
import seaborn as sns
from .drug_evaluators import (
    QED, SCScore, ESOL, SAS, Lipinski, Tox21, SIDER, ClinTox, OrganDB
)
from pytoda.transforms import LeftPadding, ToTensor
from paccmann_predictor.utils.utils import get_device
from paccmann_chemistry.utils.search import SamplingSearch


class REINFORCE(object):

    def __init__(
        self, generator, encoder, predictor, gep_df, params, model_name, logger
    ):
        """
        Constructor for the Reinforcement object.

        Args:
            generator: SMILES generator object.
            encoder: A gene expression encoder (DenseEncoder object)
            predictor: A IC50 predictor, i.e. PaccMann (MCA object).
            gep_df: A pandas df with gene expression profiles of cancer cells.
                GEP values need to be ordered, s.t. both PaccMann and encoder
                understand it.
            params: dict with hyperparameter.
            model_name: name of the model.
            logger: a logger.

        Returns:
            object of type Reinforcement used for biasing the properties
            estimated by the predictor of trajectories produced by the
            generator to maximize the custom reward function get_reward.
        """

        super(REINFORCE, self).__init__()

        self.generator = generator
        self.generator.eval()

        self.encoder = encoder
        self.encoder.eval()

        self.predictor = predictor
        self.predictor.eval()

        self.gep_df = gep_df
        self.logger = logger
        self.device = get_device()

        self.optimizer = torch.optim.Adam(
            (
                list(self.generator.decoder.parameters()) +
                list(self.encoder.parameters())
            ),
            lr=params.get('learning_rate', 0.0001),
            eps=params.get('eps', 0.0001),
            weight_decay=params.get('weight_decay', 0.00001)
        )

        self.update_params(params)
        # assert generator.decoder.latent_dim == encoder.latent_size, \
        #     'latent size of encoder and decoder do not match.'

        self.model_name = model_name
        self.model_path = os.path.join(
            params.get('model_folder', 'biased_models'), model_name
        )
        self.weights_path = os.path.join(self.model_path, 'weights/{}')

        self.smiles_to_tensor = ToTensor(self.device)
        self.pad_smiles_predictor = LeftPadding(
            params['predictor_smiles_length'],
            predictor.smiles_language.padding_index
        )

        # If model does not yet exist, create it.
        if not os.path.isdir(self.model_path):
            os.makedirs(
                os.path.join(self.model_path, 'weights'), exist_ok=True
            )
            os.makedirs(
                os.path.join(self.model_path, 'results'), exist_ok=True
            )
            # Save untrained models
            self.save('generator_epoch_0.pt', 'encoder_epoch_0.pt')

            with open(
                os.path.join(self.model_path, 'model_params.json'), 'w'
            ) as f:
                json.dump(params, f)
        else:
            self.logger.warning(
                'Model exists already. Call model.load() to restore weights.'
            )

    def update_params(self, params):
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
            'generate_len', self.predictor.params['smiles_padding_length'] - 2
        )
        # smoothing factor for softmax during token sampling in decoder
        self.temperature = params.get('temperature', 0.8)
        # critic: upper and lower bound of log(IC50) for de-normalization
        self.ic50_min = params.get('IC50_min', -8.77435)
        self.ic50_max = params.get('IC50_max', 11.83146)
        # efficacy threshold: log(X mol) = thresh. thresh=0 -> 1 micromolar
        self.ic50_threshold = params.get('IC50_threshold', 0.0)
        # rewards for efficient and all other valid molecules
        self.rewards = params.get('rewards', (11, 1))  # deprecated
        # gradient clipping in decoder
        self.grad_clipping = params.get('clip_grad', None)

    def get_log_molar(self, y):
        """
        Converts PaccMann predictions from [0,1] to log(micromolar) range.
        """
        return y * (self.ic50_max - self.ic50_min) + self.ic50_min

    def reparameterize(self, mu, logvar):
        """
        Applies reparametrization trick to obtain sample from latent space.

        Args:
            mu (torch.Tensor): The latent means of shape bs x latent_size.
            logvar (toch.Tensor): Latent log variances, shape bs x latent_size.

        Returns:
            torch.Tensor: Sampled Z from the latent distribution.
        """
        return torch.randn_like(mu).mul_(torch.exp(0.5 * logvar)).add_(mu)

    def encode_cell_line(self, cell_line=None, batch_size=128):
        """
        Encodes cell line in latent space with GEP encoder.
        """
        if cell_line is None:
            latent_z = torch.randn(1, batch_size, self.encoder.latent_size)
        else:
            gep_t = torch.unsqueeze(
                torch.Tensor(
                    self.gep_df[
                        self.gep_df['cell_line'] == cell_line  # yapf: disable
                    ].iloc[0].gene_expression.values
                ),
                0
            )
            cell_mu, cell_logvar = self.encoder(gep_t)

            latent_z = torch.unsqueeze(
                self.reparameterize(
                    cell_mu.repeat(batch_size, 1),
                    cell_logvar.repeat(batch_size, 1)
                ), 0
            )
        return latent_z

    def smiles_to_numerical(self, smiles_list, target='predictor'):
        """
        Receives a list of SMILES.
        Converts it to a numerical torch Tensor according to smiles_language
        """

        if target == 'generator':
            # NOTE: Code for this in the normal REINFORCE class
            raise ValueError('Priming drugs not yet supported')

        # Convert strings to numbers and padd length.

        smiles_num = [
            torch.unsqueeze(
                self.smiles_to_tensor(
                    self.pad_smiles_predictor(
                        self.predictor.smiles_language.
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

    def generate_compounds_and_evaluate(
        self,
        epoch,
        batch_size,
        cell_line=None,
        primed_drug=' ',
        return_latent=False,
        remove_invalid=True
    ):
        """
        Generate some compounds and evaluate them with the predictor

        Args:
            epoch (int): The training epoch.
            batch_size (int): The batch size.
            cell_line (str): A string, the cell_line used to drive generator.
            primed_drug (str): SMILES string to prime the generator.

        Returns:
            np.array: Predictions from PaccMann.
        """
        self.predictor.eval()
        self.encoder.eval()
        self.generator.eval()

        if cell_line is None:
            # Generate a random molecule
            latent_z = torch.randn(
                1, batch_size, self.generator.decoder.latent_dim
            )
        else:
            gep_t = torch.unsqueeze(
                torch.Tensor(
                    self.gep_df[
                        self.gep_df['cell_line'] == cell_line  # yapf: disable
                    ].iloc[0].gene_expression.values
                ),
                0
            )
            cell_mu, cell_logvar = self.encoder(gep_t)

            # TODO: I need to make sure that I only sample around the encoding
            # of the cell, not the entire latent space.
            latent_z = torch.unsqueeze(
                self.reparameterize(
                    cell_mu.repeat(batch_size, 1),
                    cell_logvar.repeat(batch_size, 1)
                ), 0
            )
        if type(primed_drug) != str:
            raise TypeError('Provide drug as SMILES string.')

        # # Prime the generator (optional)
        # num_drug = self.smiles_to_numerical(
        #     [primed_drug], pad_len=None, target='generator'
        # )
        # drug_mu, drug_logvar = self.generator.encode(num_drug)
        # drug_mu, drug_logvar = drug_mu[0, :], drug_logvar[0, :]
        # latent_drug = torch.unsqueeze(
        #     self.reparameterize(
        #         drug_mu.repeat(batch_size, 1),
        #         drug_logvar.repeat(batch_size, 1)
        #     ), 0
        # )
        # latent_z += latent_drug

        # Generate drugs
        valid_smiles, valid_nums, _ = self.get_smiles_from_latent(
            latent_z, remove_invalid=remove_invalid
        )

        smiles_t = self.smiles_to_numerical(valid_smiles, target='predictor')

        # Evaluate drugs
        pred, pred_dict = self.predictor(
            smiles_t, gep_t.repeat(len(valid_smiles), 1)
        )
        log_preds = self.get_log_molar(np.squeeze(pred.detach().numpy()))
        #self.plot_hist(log_preds, cell_line, epoch, batch_size)

        if return_latent:
            return valid_smiles, log_preds, latent_z
        else:
            return valid_smiles, log_preds

    def get_smiles_from_latent(self, latent, remove_invalid=True):
        """
        Takes some samples from latent space.
        Args:
            latent (torch.Tensor): tensor of shape 1 x batch_size x latent_dim.
            remove_invalid (bool): whether invalid SMILES are to be removed.
                Deaults to True.

        Returns:
            tuple(list, list): SMILES and numericals.
        """
        if self.generator.decoder.latent_dim == 2 * self.encoder.latent_size:
            latent = latent.repeat(1, 1, 2)
        mols_numerical = self.generator.generate(
            latent,
            prime_input=torch.Tensor(
                [self.generator.smiles_language.start_index]
            ).long(),
            end_token=torch.Tensor([self.generator.smiles_language.stop_index]).long(),
            generate_len=self.generate_len,
            search=SamplingSearch(temperature=self.temperature)
        )  # yapf: disable
        # Retrieve SMILES from numericals
        smiles_num_tuple = [
            (
                self.generator.smiles_language.token_indexes_to_smiles(
                    mol_num.tolist()
                ),
                torch.cat(
                    [
                        mol_num.long(),
                        torch.tensor(
                            2 * [self.generator.smiles_language.stop_index]
                        )
                    ]
                )
            ) for mol_num in iter(mols_numerical)
        ]
        numericals = [sm[1] for sm in smiles_num_tuple]

        # NOTE: If SMILES is used instead of SELFIES this line needs adjustment
        smiles = [
            self.generator.smiles_language.selfies_to_smiles(sm[0])
            for sm in smiles_num_tuple
        ]

        imgs = [Chem.MolFromSmiles(s, sanitize=True) for s in smiles]
        valid_idxs = [ind for ind in range(len(imgs)) if imgs[ind] is not None]

        smiles = [
            smiles[ind] for ind in range(len(imgs))
            if not (remove_invalid and imgs[ind] is None)
        ]
        nums = [
            numericals[ind] for ind in range(len(numericals))
            if not (remove_invalid and imgs[ind] is None)
        ]

        self.logger.info(
            f'{self.model_name}: SMILES validity: '
            f'{(len([i for i in imgs if i is not None]) / len(imgs)) * 100:.2f}%.'
        )
        return smiles, nums, valid_idxs

    def update_reward_fn(self, params):
        """ Set the reward function
        
        Arguments:
            params {dict} -- Hyperparameter for PaccMann reward function


        """
        self.paccmann_weight = params.get('paccmann_weight', 1.)
        self.qed_weight = params.get('qed_weight', 1.)
        self.scscore_weight = params.get('scscore_weight', 1.)
        self.esol_weight = params.get('esol_weight', 1.)
        self.clintox_weight = params.get('clintox_weight', 1.)
        self.organdb_weight = params.get('organdb_weight', 1.)
        self.sider_weight = params.get('sider_weight', 1.)
        self.tox21_weight = params.get('tox21_weight', 1.)

        # This is the joint reward function. Each score is normalized to be
        # inside the range [0, 1].
        # SCScore is in [1, 5] with 5 being worst
        # QED is naturally in [0, 1] with 1 being best

        self.reward_fn = (
            lambda smiles, cell: (
                self.paccmann_weight * self.get_reward_paccmann(smiles, cell) +
                np.array(
                    [
                        self.qed_weight * self.qed(s) + self.scscore_weight *
                        (self.scscore(s) - 1) * -1 / 4 + self.esol_weight *
                        (1 if self.esol(s) < -8 and self.esol(s) > -2 else 0) +
                        self.tox21_weight * self.tox21(s) + self.sider_weight *
                        self.sider(s) + self.clintox_weight * self.clintox(s) +
                        self.organdb_weight * self.organdb(s) for s in smiles
                    ]
                )
            )
        )

    def get_reward(self, valid_smiles, cell_line):
        """Get the reward
        
        Arguments:
            valid_smiles (list): A list of valid SMILES strings.
            cell_line (str): String containing the cell line to index.
        
        Returns:
            np.array: computed reward.
        """
        return self.reward_fn(valid_smiles, cell_line)

    def get_reward_paccmann(self, valid_smiles, cell_line):
        """
        Get the reward from PaccMann

        Args:
            valid_smiles (list): A list of valid SMILES strings.
            cell_line (str): String containing the cell line to index.

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

        gep_t = torch.unsqueeze(
            torch.Tensor(
                self.gep_df[
                    self.gep_df['cell_line'] == cell_line  # yapf: disable
                ].iloc[0].gene_expression.values
            ),
            0
        )
        pred, pred_dict = self.predictor(
            smiles_tensor, gep_t.repeat(len(valid_smiles), 1)
        )
        log_micromolar_pred = self.get_log_molar(
            np.squeeze(pred.detach().numpy())
        )
        return 1 / (1 + np.exp(log_micromolar_pred))

    def policy_gradient(self, cell_line, epoch, batch_size=10):
        """
        Implementation of the policy gradient algorithm.

        Args:
            cell_line (str):  cell line to generate drugs.
            epoch (int): training epoch.
            batch_size (int): batch size.

        Returns:
            tuple(float, float): total reward and total loss.
        """
        rl_loss = 0
        self.optimizer.zero_grad()

        # Encode the cell line
        latent_z = self.encode_cell_line(cell_line, batch_size)

        # Produce molecules
        valid_smiles, valid_nums, valid_idx = self.get_smiles_from_latent(
            latent_z, remove_invalid=True
        )

        # Get rewards (list, one reward for each valid smiles)
        rewards = self.get_reward(valid_smiles, cell_line)
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

        # Compute loss
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

        summed_reward = torch.mean(torch.Tensor(rewards))
        rl_loss.backward()
        if self.grad_clipping is not None:
            torch.nn.utils.clip_grad_norm_(
                list(self.generator.decoder.parameters()) +
                list(self.encoder.parameters()), self.grad_clipping
            )
        self.optimizer.step()
        return summed_reward, rl_loss.item()

    def plot_hist(self, log_preds, cell_line, epoch, batch_size):
        percentage = np.round(
            100 * (np.sum(log_preds < 0) / len(log_preds)), 1
        )
        self.logger.info(f'Percentage of effective compounds = {percentage}')
        _ = sns.kdeplot(log_preds, shade=True)
        plt.axvline(x=0)
        plt.xlabel('Predicted log(micromolar IC50)', weight='bold', size=12)
        plt.ylabel(
            f'Density of molecules (n={batch_size})', weight='bold', size=12
        )
        cl = cell_line.replace('_', '-')
        plt.title(
            f'Predicted IC50 for compounds generated against {cl}',
            weight='bold'
        )
        valid = f'{round((len(log_preds)/batch_size)*100, 1)}% SMILES validity'

        effect = f'{percentage}% compound efficacy.'
        plt.text(
            0.05, 0.9, valid, weight='bold', transform=plt.gca().transAxes
        )
        plt.text(
            0.05, 0.8, effect, weight='bold', transform=plt.gca().transAxes
        )
        plt.xlim([-7.5, 7.5])

        plt.savefig(
            self.model_path +
            f'/results/ic50_dist_ep_{epoch}_cell_{cell_line}.pdf',
            dpi=400,
            bbox_inches='tight'
        )
        plt.clf()

    def load(
        self, generator_filepath=None, encoder_filepath=None, *args, **kwargs
    ):
        """Load Model From Path."""
        if generator_filepath is not None:
            self.generator.load_model(
                self.weights_path.format(generator_filepath), *args, **kwargs
            )
        if encoder_filepath is not None:
            self.encoder.load(
                self.weights_path.format(encoder_filepath), *args, **kwargs
            )

    def save(
        self, generator_filepath=None, encoder_filepath=None, *args, **kwargs
    ):
        """Save Model to Path."""
        if generator_filepath is not None:
            self.generator.save(
                self.weights_path.format(generator_filepath), *args, **kwargs
            )
        if encoder_filepath is not None:
            self.encoder.save(
                self.weights_path.format(encoder_filepath), *args, **kwargs
            )
