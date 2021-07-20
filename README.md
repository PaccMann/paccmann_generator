[![Build Status](https://travis-ci.com/PaccMann/paccmann_generator.svg?branch=master)](https://travis-ci.com/PaccMann/paccmann_generator)
# paccmann_generator

Multimodal generative models for PaccMann<sup>RL</sup>.

`paccmann_generator` is a package for conditional molecular design, with examples of molecule generation against gene expression profiles or protein targets.
For example, see our papers:
- [_PaccMann<sup>RL</sup>: De novo generation of hit-like anticancer molecules from transcriptomic data via reinforcement learning_](https://www.cell.com/iscience/fulltext/S2589-0042(21)00237-6) (_iScience_, 2021). In there, we use methods from deep reinforcement learning to bias a molecular generator to produce molecules that exhibit low IC50 against certain cell lines (code in this repo).
- [Data-driven molecular design for discovery and synthesis of novel ligands: a case study on SARS-CoV-2](https://iopscience.iop.org/article/10.1088/2632-2153/abe808) (_Machine Learning: Science and Technology_, 2021). In there, we use the same principle to bias a molecular generator to produce molecules that have high binding affinity against certain protein targets (code in this repo).

![Graphical abstract](https://github.com/PaccMann/paccmann_generator/blob/master/assets/overview.png "Graphical abstract")


## Installation

The library itself has few dependencies (see [setup.py](setup.py)) with loose requirements.
To run the example training script we provide environment files under `examples/`.

Create a conda environment:

```sh
conda env create -f examples/IC50/conda.yml
```

Activate the environment:

```sh
conda activate paccmann_generator
```

Install in editable mode for development:

```sh
git checkout 0.0.1  # Needed only for IC50 example (for affinity example skip this line)
pip install -e .
```

## Example usage

In the `examples/IC50` directory is a training script [train_paccmann_rl.py](./examples/IC50/train_paccmann_rl.py) that makes use of `paccmann_generator`.

```console
(paccmann_generator) $ python examples/IC50/train_paccmann_rl.py -h
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

`params_filepath` could point to [examples/IC50/example_params.json](examples/IC50/example_params.json), examples for other files can be downloaded from [here](https://ibm.box.com/v/paccmann-pytoda-data).

## References

If you use `paccmann_generator` in your projects, please cite the following:

```bib
@article{born2021datadriven,
  author = {Born, Jannis and Manica, Matteo and Cadow, Joris and Markert, Greta and Mill, Nil Adell and Filipavicius, Modestas and Janakarajan, Nikita and Cardinale, Antonio and Laino, Teodoro and {Rodr{\'{i}}guez Mart{\'{i}}nez}, Mar{\'{i}}a},
  doi = {10.1088/2632-2153/abe808},
  issn = {2632-2153},
  journal = {Machine Learning: Science and Technology},
  number = {2},
  pages = {025024},
  title = {{Data-driven molecular design for discovery and synthesis of novel ligands: a case study on SARS-CoV-2}},
  url = {https://iopscience.iop.org/article/10.1088/2632-2153/abe808},
  volume = {2},
  year = {2021}
}

@article{born2021paccmannrl,
  title = {PaccMann\textsuperscript{RL}: De novo generation of hit-like anticancer molecules from transcriptomic data via reinforcement learning},
  journal = {iScience},
  volume = {24},
  number = {4},
  pages = {102269},
  year = {2021},
  issn = {2589-0042},
  doi = {https://doi.org/10.1016/j.isci.2021.102269},
  url = {https://www.cell.com/iscience/fulltext/S2589-0042(21)00237-6},
  author = {Born, Jannis and Manica, Matteo and Oskooei, Ali and Cadow, Joris and Markert, Greta and {Rodr{\'{i}}guez Mart{\'{i}}nez}, Mar{\'{i}}a}
}
```
