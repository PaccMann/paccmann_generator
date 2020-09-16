from math import ceil

import numpy as np
import pandas as pd
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl


def add_avg_profile(omics_df):
    """
    To the DF of omics data, an average profile of each cancer site is added so
    as to enable a 'precision medicine regime' in which PaccMann^RL is tuned
    on the average of all profiles of a site.
    """
    # Create and append avg cell profiles
    omics_df_n = omics_df
    for site in set(omics_df['site']):

        omics_df_n = omics_df_n.append(
            {
                'cell_line':
                    site + '_avg',
                'cosmic_id':
                    'avg',
                'histology':
                    'avg',
                'site':
                    site + '_avg',
                'gene_expression':
                    pd.Series(
                        np.mean(
                            np.stack(
                                omics_df[
                                    omics_df['site'] == site  # yapf: disable
                                ].gene_expression.values
                            ),
                            axis=0
                        )
                    )
            },
            ignore_index=True
        )

    return omics_df_n


def omics_data_splitter(omics_df, site, test_fraction):
    """
    Split omics data of cell lines into train and test.
    Args:
        omics_df    A pandas df of omics data for cancer cell lines
        site        The cancer site against which the generator is finetuned
        test_fraction  The fraction of cell lines in test data

    Returns:
        train_cell_lines, test_cell_lines (tuple): A tuple of lists with the
            cell line names used for training and testing
    """
    if site != 'all':
        cell_lines = np.array(
            list(set(omics_df[omics_df['site'] == site]['cell_line']))
        )
    else:
        cell_lines = np.array(list(set(omics_df['cell_line'])))
    inds = np.arange(cell_lines.shape[0])
    np.random.shuffle(inds)
    test_cell_lines = cell_lines[inds[:ceil(len(cell_lines) * test_fraction)]]
    train_cell_lines = cell_lines[inds[ceil(len(cell_lines) * test_fraction):]]

    return train_cell_lines.tolist(), test_cell_lines.tolist()


def disable_rdkit_logging():
    """
    Disables RDKit whiny logging.
    """
    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)
    rkrb.DisableLog('rdApp.error')
