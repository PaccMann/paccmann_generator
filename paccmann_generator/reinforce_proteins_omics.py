"""PaccMann^RL: Protein-driven drug generation"""
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from rdkit import Chem
import torch.nn.functional as F

from pytoda.transforms import LeftPadding, ToTensor
from .drug_evaluators import (
    QED, SCScore, ESOL, SAS, Lipinski, Tox21, SIDER, ClinTox, OrganDB
)
from .deepset import ravanbakhsh_set_layer
from .reinforce import Reinforce


class ReinforceProteinOmics(Reinforce):

    def __init__(
        self, generator, encoder_protein, encoder_omics, affinity_predictor, 
        efficacy_predictor, protein_df, gep_df, params, generator_smiles_language, model_name, logger, remove_invalid
    ):
        """
        Constructor for the Reinforcement object.

        Args:
            generator (nn.Module): SMILES generator object.
            encoder_protein (nn.Module): A protein encoder (DenseEncoder object)
            encoder_omics (nn.Module): A gene expression encoder (DenseEncoder object)
            affinity_predictor (nn.Module): A binding affinity predictor
            efficacy_predictor: A IC50 predictor, i.e. PaccMann (MCA object).
            protein_df (pd.Dataframe): Protein sequences of interest.
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

        super(ReinforceProteinOmics, self).__init__(
            generator, encoder_protein, params, model_name, logger, remove_invalid
        )  # yapf: disable

        self.encoder_omics = encoder_omics
        self.encoder_omics.eval()

        a= list(self.generator.decoder.parameters())
        #a.extend(list(self.encoder.encoding.parameters()))
        #a.extend(list(self.encoder_omics.encoding.parameters()))
        self.optimizer = torch.optim.Adam(
            a,
            lr=params.get('learning_rate', 0.0001),
            eps=params.get('eps', 0.0001),
            weight_decay=params.get('weight_decay', 0.00001)
        )

        self.protein_df = protein_df
        
        self.affinity_predictor = affinity_predictor
        self.affinity_predictor.eval()

        self.efficacy_predictor = efficacy_predictor
        self.efficacy_predictor.eval()

        self.gep_df = gep_df
       # self.update_params(params) # its there twice

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

        self.generator_smiles_language = generator_smiles_language
        #print(generator_smiles_language.__dict__)
        #1/0

        self.remove_invalid = remove_invalid

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
        #super().update_params(params)
        self.update_new_params(params)

         # critic: upper and lower bound of log(IC50) for de-normalization
        self.ic50_min = params.get('IC50_min', -8.77435)
        self.ic50_max = params.get('IC50_max', 11.83146)
        # efficacy threshold: log(X mol) = thresh. thresh=0 -> 1 micromolar
        self.ic50_threshold = params.get('IC50_threshold', 0.0)
        # rewards for efficient and all other valid molecules
        self.rewards = params.get('rewards', (11, 1))
        self.C_frac_weight = params.get('C_frac_weight', 0)
        self.C_frac = params.get('C_frac', 0.8)

    def update_new_params(self, params):
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
            'generate_len', self.efficacy_predictor.params['smiles_padding_length'] - 2
        )

        # smoothing factor for softmax during token sampling in decoder
        self.temperature = params.get('temperature', 0.8)
        # gradient clipping in decoder
        self.grad_clipping = params.get('clip_grad', None)

    def get_log_molar(self, y):
        """
        Converts PaccMann predictions from [0,1] to log(micromolar) range.
        """
        return y * (self.ic50_max - self.ic50_min) + self.ic50_min

    def encode_cell_line(self, cell_line=None, batch_size=128):
        """
        Encodes cell line in latent space with GEP encoder.
        """
        cell_mu = []
        cell_logvar = []
        #gep_ts = []
        for cell in cell_line:
            gep_t = torch.unsqueeze(
                torch.Tensor(
                    self.gep_df[
                        self.gep_df['cell_line'] == cell  # yapf: disable
                    ].iloc[0].gene_expression.values
                ),
                0
            )
            #gep_ts.append(torch.unsqueeze(gep_t,0).detach().numpy()[0][0])
            cell_mu_i, cell_logvar_i = self.encoder_omics(gep_t)
            cell_mu.append(torch.unsqueeze(cell_mu_i, 0).detach().numpy()[0][0])
            cell_logvar.append(torch.unsqueeze(cell_logvar_i, 0).detach().numpy()[0][0])
        #gep_ts = torch.as_tensor(gep_ts)
        cell_mu = torch.as_tensor(cell_mu)
        cell_logvar = torch.as_tensor(cell_logvar)

        cell_mu_batch = cell_mu.repeat(batch_size, 1)
        cell_logvar = cell_logvar.repeat(batch_size, 1)
        #gep_ts = gep_ts.repeat(batch_size, 1)
        if cell_mu_batch.size()[0]>batch_size:
            cell_mu_batch = cell_mu_batch[:batch_size]
            cell_logvar = cell_logvar[:batch_size]
            #gep_ts = gep_ts[:batch_size]
        latent_z = torch.unsqueeze(
            self.reparameterize(
                cell_mu_batch,
                cell_logvar
            ), 0
        )
        return latent_z

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

        if encoder_uses_sequence or predictor_uses_sequence:
            protein_sequence = self.protein_df.loc[protein]['sequence']
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
        cell_line=None,
        primed_drug=' ',
        return_latent=False
    ):
        """
        Generate some compounds and evaluate them with the predictor

        Args:
            epoch (int): The training epoch.
            batch_size (int): The batch size.
            protein (str): A string, the protein used to drive generator.
            cell_line (str): A string, the cell_line used to drive generator.
            primed_drug (str): SMILES string to prime the generator.

        Returns:
            np.array: Predictions from PaccMann.
        """
        if primed_drug != ' ':
            raise ValueError('Drug priming not yet supported.')

        self.affinity_predictor.eval()
        self.efficacy_predictor.eval()
        self.encoder.eval()
        self.encoder_omics.eval()
        self.generator.eval()

        if protein is None:
            # Generate a random molecule
            latent_z_protein = torch.randn(
                1, batch_size, self.generator.decoder.latent_dim
            )
        else:
            #protein
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
            
            latent_z_protein = torch.unsqueeze(
                self.reparameterize(
                    protein_mu_batch,
                    protein_logvar
                ), 0
            )
        if cell_line is None:
            # Generate a random molecule
            latent_z_omics = torch.randn(
                1, batch_size, self.generator.decoder.latent_dim
            )
        else:
            #cell:
            #print(cell_line)
            #print(np.sum(self.gep_df['cell_line'].isin(cell_line)), "iloc0 \n", self.gep_df[self.gep_df['cell_line'].isin(cell_line)]['gene_expression'])
            cell_mu = []
            cell_logvar = []
            gep_ts = []
            for cell in cell_line:
                gep_t = torch.unsqueeze(
                    torch.Tensor(
                        self.gep_df[
                            self.gep_df['cell_line'] == cell  # yapf: disable
                        ].iloc[0].gene_expression.values
                    ),
                    0
                )
                gep_ts.append(torch.unsqueeze(gep_t,0).detach().numpy()[0][0])
                cell_mu_i, cell_logvar_i = self.encoder_omics(gep_t)
                #print(torch.unsqueeze(cell_mu_i, 0).detach().numpy())
                cell_mu.append(torch.unsqueeze(cell_mu_i, 0).detach().numpy()[0][0])
                cell_logvar.append(torch.unsqueeze(cell_logvar_i, 0).detach().numpy()[0][0])
            gep_ts = torch.as_tensor(gep_ts)
            cell_mu = torch.as_tensor(cell_mu)
            cell_logvar = torch.as_tensor(cell_logvar)
            #print("before", cell_mu.size())
            #print(cell_mu)
            cell_mu_batch = cell_mu.repeat(batch_size, 1)
            cell_logvar = cell_logvar.repeat(batch_size, 1)
            gep_ts = gep_ts.repeat(batch_size, 1)
            if cell_mu_batch.size()[0]>batch_size:
                cell_mu_batch = cell_mu_batch[:batch_size]
                cell_logvar = cell_logvar[:batch_size]
                gep_ts = gep_ts[:batch_size]
            #print("second size:", cell_mu_batch.size(), cell_mu_batch)
            #cell_mu, cell_logvar = self.encoder_omics(gep_t)
            latent_z_omics = torch.unsqueeze(
                self.reparameterize(
                    cell_mu_batch,
                    cell_logvar
                ), 0
            )
            
        latent_z =self.together(latent_z_omics, latent_z_protein)

        # Generate drugs
        valid_smiles, valid_nums, valid_idx = self.get_smiles_from_latent(
            latent_z
        )

        smiles_t_affinity = self.smiles_to_numerical(valid_smiles, target='affinity')
        smiles_t_efficacy = self.smiles_to_numerical(valid_smiles, target='efficacy')

        # TODO: combine bowth predictors
        # Evaluate drugs
        predP, pred_dictP = self.affinity_predictor(
            smiles_t_affinity, protein_predictor_tensor[valid_idx]
        )
        predP = np.squeeze(predP.detach().numpy())
        #self.plot_hist(log_preds, cell_line, epoch, batch_size)
        # Evaluate drugs
        predO, pred_dictO = self.efficacy_predictor(
            smiles_t_efficacy, gep_ts[valid_idx]
        )
        log_predsO = self.get_log_molar(np.squeeze(predO.detach().numpy()))

        if return_latent:
            return valid_smiles, predP, log_predsO, latent_z
        else:
            return valid_smiles, predP, log_predsO, valid_idx

    def generate_compound(
        self,
        batch_size,
        protein=None,
        cell_line=None,
        primed_drug=' ',
        return_latent=False
    ):
        """
        Generate some compounds and evaluate them with the predictor

        Args:
            batch_size (int): The batch size.
            protein (str): A string, the protein used to drive generator.
            cell_line (str): A string, the cell_line used to drive generator.
            primed_drug (str): SMILES string to prime the generator.

        Returns:
            np.array: Predictions from PaccMann.
        """
        if primed_drug != ' ':
            raise ValueError('Drug priming not yet supported.')

        self.affinity_predictor.eval()
        self.efficacy_predictor.eval()
        self.encoder.eval()
        self.encoder_omics.eval()
        self.generator.eval()

        if protein is None:
            # Generate a random molecule
            latent_z_protein = torch.randn(
                1, batch_size, self.generator.decoder.latent_dim
            )
        else:
            #protein
            protein_mu = []
            protein_logvar = []
            protein_predictor_tensor = []
            for prot in protein:
                #print("protein:", prot, )
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
            
            latent_z_protein = torch.unsqueeze(
                self.reparameterize(
                    protein_mu_batch,
                    protein_logvar
                ), 0
            )
        if cell_line is None:
            # Generate a random molecule
            latent_z_omics = torch.randn(
                1, batch_size, self.generator.decoder.latent_dim
            )
        else:
            #cell:
            #print(cell_line)
            #print(np.sum(self.gep_df['cell_line'].isin(cell_line)), "iloc0 \n", self.gep_df[self.gep_df['cell_line'].isin(cell_line)]['gene_expression'])
            cell_mu = []
            cell_logvar = []
            gep_ts = []
            for cell in cell_line:
                #print("cell", cell)
                gep_t = torch.unsqueeze(
                    torch.Tensor(
                        self.gep_df[
                            self.gep_df['cell_line'] == cell  # yapf: disable
                        ].iloc[0].gene_expression.values
                    ),
                    0
                )
                gep_ts.append(torch.unsqueeze(gep_t,0).detach().numpy()[0][0])
                cell_mu_i, cell_logvar_i = self.encoder_omics(gep_t)
                #print(torch.unsqueeze(cell_mu_i, 0).detach().numpy())
                cell_mu.append(torch.unsqueeze(cell_mu_i, 0).detach().numpy()[0][0])
                cell_logvar.append(torch.unsqueeze(cell_logvar_i, 0).detach().numpy()[0][0])
            gep_ts = torch.as_tensor(gep_ts)
            cell_mu = torch.as_tensor(cell_mu)
            cell_logvar = torch.as_tensor(cell_logvar)
            #print("before", cell_mu.size())
            #print(cell_mu)
            cell_mu_batch = cell_mu.repeat(batch_size, 1)
            cell_logvar = cell_logvar.repeat(batch_size, 1)
            gep_ts = gep_ts.repeat(batch_size, 1)
            if cell_mu_batch.size()[0]>batch_size:
                cell_mu_batch = cell_mu_batch[:batch_size]
                cell_logvar = cell_logvar[:batch_size]
                gep_ts = gep_ts[:batch_size]
            #print("second size:", cell_mu_batch.size(), cell_mu_batch)
            #cell_mu, cell_logvar = self.encoder_omics(gep_t)
            latent_z_omics = torch.unsqueeze(
                self.reparameterize(
                    cell_mu_batch,
                    cell_logvar
                ), 0
            )
            
        latent_z =self.together(latent_z_omics, latent_z_protein)

        # Generate drugs
        valid_smiles, valid_nums, valid_idx = self.get_smiles_from_latent(
            latent_z
        )
        return valid_smiles, valid_idx

    def update_reward_fn(self, params):
        """ Set the reward function
        
        Arguments:
            params {dict} -- Hyperparameter for PaccMann reward function


        """
        super().update_reward_fn(params)
        self.paccmann_weight = params.get('paccmann_weight', 1.)
        self.affinity_weight = params.get('affinity_weight', 1.)
        self.C_frac_weight = params.get('C_frac_weight', 0)
        self.weight_tot = self.paccmann_weight + self.affinity_weight + self.tox21_weight + self.qed_weight + self.scscore_weight + self.esol_weight
        
        def tox_f(s):
            x = 0
            if self.tox21_weight > 0.:
                x += self.tox21_weight * self.tox21(s)
            if self.sider_weight > 0.:
                x += self.sider_weight * self.sider(s)
                self.weight_tot += self.sider_weight
            if self.clintox_weight > 0.:
                x += self.clintox_weight * self.clintox(s)
                self.weight_tot += self.clintox_weight
            if self.organdb_weight > 0.:
                x += self.organdb_weight * self.organdb(s)
                self.weight_tot += self.organdb_weight
            if self.C_frac_weight > 0.:
                x += self.C_frac_weight * self.get_C_fraction(s)
                self.weight_tot += -np.absolut(self.C_frac - self.C_frac_weight)
            return x

        # This is the joint reward function. Each score is normalized to be
        # inside the range [0, 1].
        self.reward_fn = (
            lambda smiles, protein, cell, idx, batch_size: (
                self.affinity_weight / self.weight_tot * self.get_reward_affinity(smiles, protein, idx, batch_size) + 
                np.array([tox_f(s) for s in smiles]) +
                self.paccmann_weight / self.weight_tot * self.get_reward_paccmann(smiles, cell, idx, batch_size) +
                np.array(
                    [
                        self.qed_weight / self.weight_tot * self.qed(s) + 
                        self.scscore_weight / self.weight_tot *((self.scscore(s) - 1) * (-1 / 4) + 1) + 
                        self.esol_weight / self.weight_tot * (1 if self.esol(s) > -8 and self.esol(s) < -2 else 0) +
                        #self.tox21_weight * self.tox21(s) + 
                        #self.sider_weight * self.sider(s) + 
                        #self.clintox_weight * self.clintox(s) +
                        #self.organdb_weight * self.organdb(s) for s in smiles
                        tox_f(s) for s in smiles
                    ]
                    # minimize the difference of fraction of C to C_frac
                )
            )
        )

    def get_C_fraction(self, smiles):
        """get the fraction of C atoms in the molecule

        Args:
            smiles (list): A list of SMILES strings.

        Returns:
            list: a list of the fractions of C atmons per molecule.
        """
        tot = []
        if smiles:
            for s in smiles:
                if len(s) is not 0:
                    C = [1 for i in s if i=='C'].count(1)
                    tot.append(C/len(s)) #get sometimes a div by 0 error
        return tot

    def get_reward(self, valid_smiles, protein, cell_line, idx, batch_size):
        """Get the reward
        
        Arguments:
            valid_smiles (list): A list of valid SMILES strings.
            protein (str): Name of the target protein.
            cell_line (str): String containing the cell line to index.
        
        Returns:
            np.array: computed reward.
        """
        return self.reward_fn(valid_smiles, protein, cell_line, idx, batch_size)

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
            valid_smiles, target='affinity'
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
        #if self.remove_invalid:
        #    smiles_list_new = []
        #    for s in smiles_list:
        #        try:
        #            smiles_list_new.append(Chem.MolToSmiles(
        #                    Chem.MolFromSmiles(s, sanitize=True), kekuleSmiles=True
        #            ).replace(':', ''))
        #        except:
        #            print("error occured in smiles", s)
        #            smiles_list_new.append('')
        #else:
        #    smiles_list_new = []
        #    for s in smiles_list:
        #        try:
        #            smiles_list_new.append(Chem.MolToSmiles(
        #                    Chem.MolFromSmiles(s, sanitize=False), kekuleSmiles=True
        #            ).replace(':', ''))
        #        except:
        #            print("error occured in smiles", s)
        #            smiles_list_new.append(s)
        #smiles_list = smiles_list_new

        if target == 'efficacy':
            # Convert strings to numbers and padd length.
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
            # Convert strings to numbers and padd length.
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

        # Catch scenario where all SMILES are invalid
        if len(smiles_num) == 0:
            return torch.Tensor()

        smiles_tensor = torch.cat(smiles_num, dim=0)
        if target == 'efficacy':
            smiles_tensor = smiles_tensor.narrow(1, len(smiles_tensor[0])-250, 250)

        return smiles_tensor
    
    def get_reward_paccmann(self, valid_smiles, cell_line, idx, batch_size):
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
            valid_smiles, target='efficacy'
        )

        # If all SMILES are invalid, no reward is given
        if len(smiles_tensor) == 0:
            return 0

        gep_ts = []
        for cell in cell_line:
            gep_t = torch.unsqueeze(
                torch.Tensor(
                    self.gep_df[
                        self.gep_df['cell_line'] == cell  # yapf: disable
                    ].iloc[0].gene_expression.values
                ),
                0
            )
            gep_ts.append(torch.unsqueeze(gep_t,0).detach().numpy()[0][0])
        gep_ts = torch.as_tensor(gep_ts)
        gep_ts = gep_ts.repeat(batch_size, 1)
        if gep_ts.size()[0]>batch_size:
            gep_ts = gep_ts[:batch_size]

        pred, pred_dict = self.efficacy_predictor(
            smiles_tensor, gep_ts[idx]
        )
        log_micromolar_pred = self.get_log_molar(
            np.squeeze(pred.detach().numpy())
        )
        lmps = [
            lmp if lmp < self.ic50_threshold else 10
            for lmp in log_micromolar_pred
        ]
        return 1 / (1 + np.exp(lmps))

    def together(self, latent_z_omics, latent_z_protein):
        return self.together_mean(latent_z_omics, latent_z_protein)

    def together_mean(self, latent_z_omics, latent_z_protein):
        return torch.mean(torch.cat((latent_z_protein, latent_z_omics),0),0)

    def together_concat(self, latent_z_omics, latent_z_protein):
        latent_z_long = torch.cat((latent_z_protein, latent_z_omics),2).transpose(2,1)
        latent_z = ravanbakhsh_set_layer(latent_z_omics.size()[2], latent_z_long).transpose(2,1)
        return latent_z

    def policy_gradient(self, protein, cell_line, epoch, batch_size=10):
        """
        Implementation of the policy gradient algorithm.

        Args:
            protein (str): Name of protein to generate drugs.
            cell_line (str):  cell line to generate drugs.
            epoch (int): training epoch.
            batch_size (int): batch size.

        Returns:
            tuple(float, float): total reward and total loss.
        """
        rl_loss = 0
        self.optimizer.zero_grad()

        # Encode the protein
        latent_z_protein = self.encode_protein(protein, batch_size)
        
        # Encode the cell line
        latent_z_omics = self.encode_cell_line(cell_line, batch_size)

        latent_z = self.together(latent_z_omics, latent_z_protein)

        # Produce molecules
        valid_smiles, valid_nums, valid_idx = self.get_smiles_from_latent(
            latent_z
        )

        # Get rewards (list, one reward for each valid smiles)
        rewards = self.get_reward(valid_smiles, protein, cell_line, valid_idx, batch_size)
        reward_tensor = torch.unsqueeze(torch.Tensor(rewards), 1)

        # valid_nums is a list of torch.Tensor, each with varying length,
        padded_nums = torch.nn.utils.rnn.pad_sequence(valid_nums)
        num_mols = padded_nums.shape[1]

        self.generator.decoder._update_batch_size(num_mols)

        # Batch processing
        lrps = 1
        if self.generator.decoder.latent_dim == 2 * self.encoder_omics.latent_size:
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
                list(self.generator.decoder.parameters()) + list(self.encoder.parameters()) +
                list(self.encoder_omics.parameters()), self.grad_clipping
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
        self, generator_filepath=None, encoder_filepath_protein=None, encoder_filepath_omics=None, *args, **kwargs
    ):
        """Load Model From Path."""
        if generator_filepath is not None:
            self.generator.load(
                self.weights_path.format(generator_filepath), *args, **kwargs
            )
        if encoder_filepath_protein is not None:
            self.encoder.load(
                self.weights_path.format(encoder_filepath_protein), *args, **kwargs
            )
        if encoder_filepath_omics is not None:
            self.encoder_omics.load(
                self.weights_path.format(encoder_filepath_omics), *args, **kwargs
            )

    def save(
        self, generator_filepath=None, encoder_filepath_protein=None, encoder_filepath_omics=None, *args, **kwargs
    ):
        """Save Model to Path."""
        if generator_filepath is not None:
            self.generator.save(
                self.weights_path.format(generator_filepath), *args, **kwargs
            )
        if encoder_filepath_omics is not None:
            self.encoder_omics.save(
                self.weights_path.format(encoder_filepath_omics), *args, **kwargs
            )
        if encoder_filepath_protein is not None:
            self.encoder.save(
                self.weights_path.format(encoder_filepath_protein), *args, **kwargs
            )