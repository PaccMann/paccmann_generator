"""PaccMann^RL: Policy gradient class (implemented through REINFORCE)."""
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
import matplotlib.pyplot as plt
import seaborn as sns
from pytoda.transforms import LeftPadding, ToTensor
from .drug_evaluators import QED, SCScore, ESOL, SAS, Lipinski
from paccmann_predictor.utils.utils import get_device


class REINFORCE_proteins(object):

    def __init__(
        self, generator, encoder, predictor, protein_df, smiles_language,
        protein_language, params, model_name, logger
    ):
        """
        Constructor for the Reinforcement object.

        Args:
            generator: SMILES generator object.
            encoder: A gene expression encoder (DenseEncoder object)
            predictor: A IC50 predictor, i.e. PaccMann (MCA object).
            protein_df (pd.Dataframe): Protein sequences of interest.
            smiles_language: A smiles_language object. Both, predictor and
                generator need to know this syntax.
            protein_language: A protein_language object for the encoder
            params: dict with hyperparameter.
            model_name: name of the model.
            logger: a logger.

        Returns:
            object of type Reinforcement used for biasing the properties
            estimated by the predictor of trajectories produced by the
            generator to maximize the custom reward function get_reward.
        """

        super(REINFORCE_proteins, self).__init__()

        self.generator = generator
        self.generator.eval()

        self.encoder = encoder
        self.encoder.eval()

        self.predictor = predictor
        self.predictor.eval()

        self.device = get_device()

        assert generator.decoder.latent_dim == encoder.latent_size, \
            'latent size of encoder and decoder do not match.'

        self.protein_df = protein_df
        self.smiles_language = smiles_language
        self.protein_language = protein_language

        assert (
            predictor.protein_padding_length == encoder.protein_padding_length
        ), (
            'Predictor and encoder need to have same padding length, found '
            f'Predictor: {predictor.protein_padding_length} and encoder:'
            f'{encoder.protein_padding_length}.'
        )

        self.pad_smiles = LeftPadding(
            predictor.smiles_padding_length, smiles_language.padding_index
        )
        self.pad_protein = LeftPadding(
            predictor.protein_padding_length, protein_language.padding_index
        )

        self.protein_to_tensor = ToTensor(self.device)
        self.smiles_to_tensor = ToTensor(self.device)
        # self.smiles_to_num = SMILESToTokenIndexes(self.smiles_language)
        self.logger = logger

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
        self.model_name = model_name
        self.model_path = os.path.join(
            params.get('model_folder', 'biased_models'), model_name
        )
        self.weights_path = os.path.join(self.model_path, 'weights/{}')

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
        self.update_reward_fn(params)
        # discount factor
        self.gamma = params.get('gamma', 0.99)
        # maximal length of generated molecules
        self.generate_len = params.get(
            'generate_len', self.predictor.params['smiles_padding_length'] - 2
        )
        # smoothing factor for softmax during token sampling in decoder
        self.temperature = params.get('temperature', 0.8)

        # gradient clipping in decoder
        self.grad_clipping = params.get('clip_grad', None)

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

            protein_tensor = self.protein_to_numerical(protein)
            protein_mu, protein_logvar = self.encoder(protein_tensor)

            latent_z = torch.unsqueeze(
                self.reparameterize(
                    protein_mu.repeat(batch_size, 1),
                    protein_logvar.repeat(batch_size, 1)
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
                    self.pad_smiles(
                        self.smiles_language.smiles_to_token_indexes(smiles)
                    )
                ), 0
            ) for smiles in smiles_list
        ]

        # Catch scenario where all SMILES are invalid
        if len(smiles_num) == 0:
            return torch.Tensor()

        smiles_tensor = torch.cat(smiles_num, dim=0)
        return smiles_tensor

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

        if encoder_uses_sequence or predictor_uses_sequence:
            protein_sequence = self.protein_df.loc[protein]['Sequence']
            sequence_tensor = torch.unsqueeze(
                self.protein_to_tensor(
                    self.pad_protein(
                        self.protein_language.
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
        t1 = sequence_tensor if encoder_uses_sequence else encoding_tensor
        t2 = sequence_tensor if predictor_uses_sequence else encoding_tensor
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

        self.predictor.eval()
        self.encoder.eval()
        self.generator.eval()

        if protein is None:
            # Generate a random molecule
            latent_z = torch.randn(
                1, batch_size, self.generator.decoder.latent_dim
            )
        else:
            protein_encoder_tensor, protein_predictor_tensor = (
                self.protein_to_numerical(protein)
            )
            protein_mu, protein_logvar = self.encoder(protein_encoder_tensor)

            # TODO: I need to make sure that I only sample around the encoding
            # of the protein, not the entire latent space.
            latent_z = torch.unsqueeze(
                self.reparameterize(
                    protein_mu.repeat(batch_size, 1),
                    protein_logvar.repeat(batch_size, 1)
                ), 0
            )

        # Generate drugs
        valid_smiles, valid_nums = self.get_smiles_from_latent(
            latent_z, remove_invalid=remove_invalid
        )

        smiles_t = self.smiles_to_numerical(valid_smiles, target='predictor')

        # Evaluate drugs
        pred, pred_dict = self.predictor(
            smiles_t, protein_predictor_tensor.repeat(len(valid_smiles), 1)
        )
        pred = np.squeeze(pred.detach().numpy())
        #self.plot_hist(log_preds, cell_line, epoch, batch_size)

        if return_latent:
            return valid_smiles, pred, latent_z
        else:
            return valid_smiles, pred

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
        mols_numerical = self.generator.generate(
            latent,
            prime_input=torch.Tensor(
                [self.smiles_language.start_index]
            ).long(),
            end_token=torch.Tensor([self.smiles_language.stop_index]).long(),
            generate_len=self.generate_len,
            temperature=self.temperature
        )  # yapf: disable
        # Retrieve SMILES from numericals
        smiles_num_tuple = [
            (
                self.smiles_language.token_indexes_to_smiles(mol_num.tolist()),
                torch.cat(
                    [
                        mol_num.long(),
                        torch.tensor(2 * [self.smiles_language.stop_index])
                    ]
                )
            ) for mol_num in iter(mols_numerical)
        ]
        smiles = [sm[0] for sm in smiles_num_tuple]
        numericals = [sm[1] for sm in smiles_num_tuple]

        imgs = [Chem.MolFromSmiles(s, sanitize=True) for s in smiles]

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
        return smiles, nums

    def update_reward_fn(self, params):
        """ Set the reward function
        
        Arguments:
            params {dict} -- Hyperparameter for PaccMann reward function


        """
        self.affinity_weight = params.get('affinity_weight', 1.)
        self.qed_weight = params.get('qed_weight', 0.)
        self.scscore_weight = params.get('scscore_weight', 0.)
        self.esol_weight = params.get('esol_weight', 0.)

        # This is the joint reward function. Each score is normalized to be
        # inside the range [0, 1].
        # SCScore is in [1, 5] with 5 being worst
        # QED is naturally in [0, 1] with 1 being best

        self.reward_fn = (
            lambda smiles, protein: (
                self.affinity_weight * self.
                get_reward_affinity(smiles, protein) + np.array(
                    [
                        self.qed_weight * self.qed(s) + self.scscore_weight *
                        (self.scscore(s) - 1) * -1 / 4 + self.esol_weight *
                        (1 if self.esol(s) < -8 and self.esol(s) > -2 else 0)
                        for s in smiles
                    ]
                )
            )
        )

    def get_reward(self, valid_smiles, protein):
        """Get the reward
        
        Arguments:
            valid_smiles (list): A list of valid SMILES strings.
            protein (str): Name of the target protein
        
        Returns:
            np.array: computed reward.
        """
        return self.reward_fn(valid_smiles, protein)

    def get_reward_affinity(self, valid_smiles, protein):
        """
        Get the reward from PaccMann

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

        protein_tensor = self.protein_to_numerical(protein)

        pred, pred_dict = self.predictor(
            smiles_tensor, protein_tensor.repeat(smiles_tensor.shape[0], 1)
        )

        return np.squeeze(pred.detach().numpy())

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
        valid_smiles, valid_nums = self.get_smiles_from_latent(
            latent_z, remove_invalid=True
        )

        # Get rewards (list, one reward for each valid smiles)
        rewards = self.get_reward(valid_smiles, protein)
        reward_tensor = torch.unsqueeze(torch.Tensor(rewards), 1)

        # valid_nums is a list of torch.Tensor, each with varying length,
        padded_nums = torch.nn.utils.rnn.pad_sequence(valid_nums)
        num_mols = padded_nums.shape[1]

        # Batch processing
        hidden = self.generator.decoder.init_hidden(num_mols)
        stack = self.generator.decoder.init_stack(num_mols)

        # # Compute loss
        for p in range(len(padded_nums) - 1):

            output, hidden, stack = self.generator.decoder(
                padded_nums[p], hidden, stack
            )
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
            self.generator.save_model(
                self.weights_path.format(generator_filepath), *args, **kwargs
            )
        if encoder_filepath is not None:
            self.encoder.save(
                self.weights_path.format(encoder_filepath), *args, **kwargs
            )
