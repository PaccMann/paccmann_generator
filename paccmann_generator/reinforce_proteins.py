"""PaccMann^RL: Protein-driven drug generation"""
import numpy as np
import torch
import torch.nn.functional as F

from pytoda.transforms import LeftPadding, ToTensor

from .reinforce import Reinforce


class ReinforceProtein(Reinforce):

    def __init__(
        self, generator, encoder, predictor, protein_df, params, model_name,
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

        self.predictor = predictor
        self.predictor.eval()

        self.protein_df = protein_df

        self.pad_smiles_predictor = LeftPadding(
            predictor.smiles_padding_length,
            predictor.smiles_language.padding_index
        )
        self.pad_protein_predictor = LeftPadding(
            predictor.protein_padding_length,
            predictor.protein_language.padding_index
        )

        self.protein_to_tensor = ToTensor(self.device)
        self.update_params(params)

    def update_params(self, params):
        """Update parameter

        Args:
            params (dict): Dict with (new) parameters
        """
        super().update_params(params)

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

            protein_tensor, _ = self.protein_to_numerical(
                protein, encoder_uses_sequence=False
            )
            protein_mu, protein_logvar = self.encoder(protein_tensor)

            latent_z = torch.unsqueeze(
                self.reparameterize(
                    protein_mu.repeat(batch_size, 1),
                    protein_logvar.repeat(batch_size, 1)
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

        if predictor_uses_sequence:
            protein_sequence = self.protein_df.loc[protein]['Sequence']
            sequence_tensor_p = torch.unsqueeze(
                self.protein_to_tensor(
                    self.pad_protein_predictor(
                        self.predictor.protein_language.
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
                self.protein_to_numerical(
                    protein, encoder_uses_sequence=False
                )
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
        valid_smiles, valid_nums, _ = self.get_smiles_from_latent(
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

    def update_reward_fn(self, params):
        """ Set the reward function
        Arguments:
            params (dict): Hyperparameter for PaccMann reward function
        """
        super().update_reward_fn(params)
        self.affinity_weight = params.get('affinity_weight', 1.)

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
            lambda smiles, protein: (
                self.affinity_weight * self.get_reward_affinity(
                    smiles, protein
                ) + np.array([tox_f(s) for s in smiles])
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

        _, protein_tensor = self.protein_to_numerical(
            protein, encoder_uses_sequence=False
        )

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
        valid_smiles, valid_nums, valid_idx = self.get_smiles_from_latent(
            latent_z, remove_invalid=True
        )

        # Get rewards (list, one reward for each valid smiles)
        rewards = self.get_reward(valid_smiles, protein)
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
