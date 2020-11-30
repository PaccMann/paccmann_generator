"""PaccMann^RL: Policy gradient class (implemented through REINFORCE)."""
import json
import os
import torch
from rdkit import Chem
from .drug_evaluators import (
    QED, SCScore, ESOL, SAS, Lipinski, Tox21, SIDER, ClinTox, OrganDB
)
from pytoda.transforms import ToTensor
from paccmann_predictor.utils.utils import get_device
from paccmann_chemistry.utils.search import SamplingSearch


class Reinforce(object):

    def __init__(self, generator, encoder, params, model_name, logger, remove_invalid):
        """
        Constructor for the Reinforcement object.

        Args:
            generator (nn.Module): SMILES generator object.
            encoder (nn.Module): An encoder object.
            params (dict): dict with hyperparameter.
            model_name (str): name of the model.
            logger: a logger.

        Returns:
            object of type REINFORCE used for biasing the properties
            estimated by the predictor of trajectories produced by the
            generator to maximize the custom reward function get_reward.
        """

        super(Reinforce, self).__init__()

        self.generator = generator
        self.generator.eval()

        self.encoder = encoder
        self.encoder.eval()

        self.logger = logger
        self.device = get_device()

        a= list(self.generator.decoder.parameters())
        #a.extend(list(self.encoder.encoding.parameters()))
        self.optimizer = torch.optim.Adam(
            a,
            lr=params.get('learning_rate', 0.0001),
            eps=params.get('eps', 0.0001),
            weight_decay=params.get('weight_decay', 0.00001)
        )

        self.model_name = model_name
        self.model_path = os.path.join(
            params.get('model_folder', 'biased_models'), model_name
        )
        self.weights_path = os.path.join(self.model_path, 'weights/{}')

        self.smiles_to_tensor = ToTensor(self.device)

        self.remove_invalid = remove_invalid

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
        return mu
        # return torch.randn_like(mu).mul_(torch.exp(0.5 * logvar)).add_(mu)

    def smiles_to_numerical(self, smiles_list, target='predictor'):
        """
        Receives a list of SMILES.
        Converts it to a numerical torch Tensor according to smiles_language
        """
        if target == 'generator':
            # NOTE: Code for this in the normal REINFORCE class
            raise ValueError('Priming drugs not yet supported')

        # # TODO: Workaround since predictor does not understand aromatic carbons
        # smiles_list = [
        #     Chem.MolToSmiles(
        #         Chem.MolFromSmiles(s, sanitize=False), kekuleSmiles=True
        #     ).replace(':', '') for s in smiles_list
        # ]

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

    def get_smiles_from_latent(self, latent):
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
                [self.generator_smiles_language.start_index]    ########################### generator.smiles_language
            ).long(),
            end_token=torch.Tensor([self.generator_smiles_language.stop_index]).long(),
            generate_len=self.generate_len,
            search=SamplingSearch(temperature=self.temperature)
        )  # yapf: disable
        # Retrieve SMILES from numericals
        smiles_num_tuple = [
            (
                self.generator_smiles_language.token_indexes_to_smiles(
                    mol_num.tolist()
                ),
                torch.cat(
                    [
                        mol_num.long(),
                        torch.tensor(
                            2 * [self.generator_smiles_language.stop_index]
                        )
                    ]
                )
            ) for mol_num in iter(mols_numerical)
        ]
        numericals = [sm[1] for sm in smiles_num_tuple]

        # NOTE: If SMILES is used instead of SELFIES this line needs adjustment
        smiles = [
            self.generator_smiles_language.selfies_to_smiles(sm[0])
            for sm in smiles_num_tuple
        ]
        if self.remove_invalid:
            imgs = [Chem.MolFromSmiles(s, sanitize=True) for s in smiles]
        else:
            imgs = [Chem.MolFromSmiles(s, sanitize=False) for s in smiles]
        valid_idxs = [ind for ind in range(len(imgs)) if (imgs[ind] is not None) and (not imgs[ind])]
        # valid_idxs = [ind for ind, img in enumerate(imgs) if img is not None]

        
        smiles, nums = zip(*[
           (smiles[ind], numericals[ind]) for ind in valid_idxs
        ])
        # nums = [
        #     numericals[ind] for ind in range(len(numericals))
        #     if not (self.remove_invalid and imgs[ind] is None)
        # ]

        self.logger.info(
            f'{self.model_name}: SMILES validity: '
            f'{(len([i for i in imgs if i is not None]) / len(imgs)) * 100:.2f}%.'
        )
        return list(smiles), list(nums), valid_idxs

    def update_reward_fn(self, params):
        """ Set the reward function
        Arguments:
            params (dict): Hyperparameter for PaccMann reward function


        """
        self.clintox_weight = params.get('clintox_weight', 0.)
        self.organdb_weight = params.get('organdb_weight', 0.)
        self.sider_weight = params.get('sider_weight', 0.)
        self.qed_weight = params.get('qed_weight', 0.)
        self.scscore_weight = params.get('scscore_weight', 0.)
        self.esol_weight = params.get('esol_weight', 0.)

        self.tox21_weight = params.get('tox21_weight', .5)
        if self.tox21_weight > 0.:
            self.tox21 = Tox21(
                params.get(
                    'tox21_path',
                    os.path.join('..', 'data', 'models', 'Tox21')
                )
            )

        self.organdb_weight = params.get('organdb_weight', 0.)
        if self.organdb_weight > 0.:
            self.organdb = OrganDB(
                params.get(
                    'organdb_path',
                    os.path.join('..', 'data', 'models', 'OrganDB')
                ), params['site']
            )

        self.clintox_weight = params.get('clintox_weight', 0.)
        if self.clintox_weight > 0.:
            self.clintox = ClinTox(
                params.get(
                    'clintox_path',
                    os.path.join('..', 'data', 'models', 'ClinTox')
                )
            )

        self.sider_weight = params.get('sider_weight', 0.)
        if self.sider_weight > 0.:
            self.sider = SIDER(
                params.get(
                    'sider_path',
                    os.path.join('..', 'data', 'models', 'Siders')
                )
            )

    def load(
        self, generator_filepath=None, encoder_filepath=None, *args, **kwargs
    ):
        """Load Model From Path."""
        if generator_filepath is not None:
            self.generator.load(
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
