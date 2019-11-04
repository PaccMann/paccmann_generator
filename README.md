[![Build Status](https://travis-ci.org/PaccMann/paccmann_generator.svg?branch=master)](https://travis-ci.org/PaccMann/paccmann_generator)
# paccmann_generator

Multimodal generative models for PaccMann^RL.

## Requirements

- `conda>=3.7`

## Installation

The library itself has few dependencies (see [setup.py](setup.py)) with loose requirements.

Create a conda environment:

```sh
conda env create -f conda.yml
```

Activate the environment:

```sh
conda activate paccmann_generator
```

Install in editable mode for development:

```sh
pip install -e .
```

To run the example training script we provide additional requirements under `examples/`.
Instal them:

```sh
pip install -r examples/requirements.txt
```

## Example usage

In the `examples` directory is a training script [train_paccmann_rl.py](./examples/train_paccmann_rl.py) that makes use of `paccmann_generator`.

```console
(paccmann_generator) $ python examples/train_paccmann_rl.py -h
usage: train_paccmann_rl.py [-h]
                            mol_model_path omics_model_path ic50_model_path
                            smiles_language_path omics_data_path params_path
                            model_name site

PaccMann^RL training script

positional arguments:
  mol_model_path        Path to chemistry model
  omics_model_path      Path to omics model
  ic50_model_path       Path to pretrained ic50 model
  smiles_language_path  Path to SMILES language object
  omics_data_path       Omics data path to condition generation
  params_path           Model params json file directory
  model_name            Name for the trained model.
  site                  Name of the cancer site for conditioning.

optional arguments:
  -h, --help            show this help message and exit
```

`params_filepath` could point to [examples/example_params.json](examples/example_params.json), examples for other files can be downloaded from [here](https://ibm.box.com/v/paccmann-pytoda-data).

## References

If you use `paccmann_generator` in your projects, please cite the following:

```bib
@misc{born2019reinforcement,
    title={Reinforcement learning-driven de-novo design of anticancer compounds conditioned on biomolecular profiles},
    author={Jannis Born and Matteo Manica and Ali Oskooei and Maria Rodriguez Martinez},
    year={2019},
    eprint={1909.05114},
    archivePrefix={arXiv},
    primaryClass={q-bio.BM}
}
```
