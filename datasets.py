import argparse
import json
import logging
import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from paccmann_chemistry.models import (
    StackGRUDecoder, StackGRUEncoder, TeacherVAE
)
from paccmann_chemistry.utils import get_device
from paccmann_generator.plot_utils import plot_and_compare, plot_and_compare_proteins, plot_loss
from paccmann_generator.utils import add_avg_profile, omics_data_splitter, protein_data_splitter
from paccmann_omics.encoders import ENCODER_FACTORY
from pytoda.smiles.smiles_language import SMILESLanguage
from pytoda.proteins.protein_language import ProteinLanguage
from paccmann_generator.utils import disable_rdkit_logging
from paccmann_predictor.models import MODEL_FACTORY as MODEL_FACTORY_OMICS
import sys
#sys.path.append('/dataP/tol/paccmann_affinity')
sys.path.append('/home/tol/paccmann_affinity')
from paccmann_affinity.models.predictors import MODEL_FACTORY as MODEL_FACTORY_PROTEIN

from paccmann_generator.reinforce_proteins_omics import ReinforceProteinOmics
from paccmann_generator import ReinforceOmic
from paccmann_generator.reinforce_proteins import ReinforceProtein
#cancer_cell_lines = ['HUH-6-clone5','HuH-7','SNU-475','SNU-423','SNU-387','SNU-449','HLE','C3A']
cancertype = 'hepatoblastoma'
site = 'all'
medulloblastoma_genes = ['MEGF8','INVS','FRAT2','FZD7','RSPO3','SFRP5','NLK','SPOPL','PRKCA','NKD2','AXIN1','RHOA','WNT10A','FZD8','TCF7','FRAT1','CHD8','NFATC4','KIF7','RSPO4','ROR2','PTCH1','AXIN2','SMURF2','TPTEP2-CSNK1E','CXXC4','WNT9A','SMAD4','PRICKLE2','RAC1','GLI2','FZD5','SPOP','PPP3CC','RBX1','CAMK2B','DKK1','CTNND2','PRKACB','VANGL2','PLCB4','WNT5B','GLI1','JUN','PPP3R2','RAC3','NFATC2','LRP5','SFRP1','WNT3A','RSPO2','WNT8A','FZD1','CDON','CACYBP','RNF43','GRK2','ZNRF3','CSNK2A3','LEF1','NFATC1','TBL1X','DKK2','CTNNBIP1','CSNK1A1L','WNT5A','PRICKLE4','CSNK2B','CCN4','SHH','KIF3A','FZD4','CREBBP','CSNK1G1','PRICKLE3','NKD1','PPP3CA','CSNK1G2','APC','WNT8B','GPC4','DAAM1','CBY1','WNT6','CSNK1E','MMP7','ROR1','WNT7B','SMURF1','MAPK8','PRKACA','CSNK2A1','EP300','VANGL1','PORCN','RAC2','BTRC','SERPINF1','WNT2','RUVBL1','TBL1XR1','CER1','SFRP4','CTNNB1','ARRB2','MGRN1','CSNK1A1','CUL3','DVL1','PPP3R1','WNT11','GPR161','TP53','NFATC3','ARRB1','SENP2','PSEN1','PRKCB','GRK3','PLCB2','TCF7L1','BAMBI','GLI3','WNT9B','FZD6','PRKCG','SUFU','MOSMO','NOTUM','FZD10','PRICKLE1','EVC','TBL1Y','FZD9','SFRP2','CCND3','GAS1','WNT3','WNT4','SKP1','PLCB1','WNT10B','DVL3','SIAH1','CSNK1D','CCND2','ROCK2','FZD3','LRP6','EVC2','WNT1','DHH','IHH','CSNK1G3','LGR6','WNT16','BCL2','MYC','APC2','MAPK10','CAMK2D','HHIP','PRKACG','PPP3CB','PTCH2','CCND1','FZD2','MAPK9','FBXW11','TCF7L2','CAMK2A','GSK3B','SMO','DVL2','CUL1','DAAM2','WIF1','DKK4','RYK','PPARD','RSPO1','FOSL1','CAMK2G','WNT2B','BOC','CTBP1','SOX17','SMAD3','LGR4','LRP2','CTBP2','LGR5','SOST','PLCB3','WNT7A','CSNK2A2','MAP3K7']
hepatoblastoma_genes = ['PORCN','WNT1','WNT2','WNT2B','WNT3','WNT3A','WNT4','WNT5A','WNT5B','WNT6','WNT7A','WNT7B','WNT8A','WNT8B','WNT9A','WNT9B','WNT10B','WNT10A','WNT11','WNT16','CER1','NOTUM','WIF1','SERPINF1','SOST','DKK1','DKK2','DKK4','SFRP1','SFRP2','SFRP4','SFRP5','RSPO1','RSPO2','RSPO3','RSPO4','LGR4','LGR5','LGR6','RNF43','ZNRF3','FZD1','FZD7','FZD2','FZD3','FZD4','FZD5','FZD8','FZD6','FZD10','FZD9','LRP5','LRP6','BAMBI','CSNK1E','TPTEP2-CSNK1E','DVL3','DVL2','DVL1','FRAT1','FRAT2','CSNK2A1','CSNK2A2','CSNK2A3','CSNK2B','NKD1','NKD2','CXXC4','SENP2','GSK3B','CTNNB1','AXIN1','AXIN2','APC','APC2','CSNK1A1L','CSNK1A1','TCF7','TCF7L1','TCF7L2','LEF1','CTNNBIP1','CBY1','CHD8','SOX17','CTBP1','CTBP2','CTNND2','CREBBP','EP300','RUVBL1','SMAD4','SMAD3','MAP3K7','NLK','MYC','JUN','FOSL1','CCND1','CCND2','CCND3','CCN4','PPARD','MMP7','PSEN1','PRKACA','PRKACB','PRKACG','TP53','SIAH1','CACYBP','SKP1','TBL1X','TBL1Y','TBL1XR1','BTRC','FBXW11','CUL1','RBX1','GPC4','ROR1','ROR2','RYK','VANGL2','VANGL1','PRICKLE1','PRICKLE2','PRICKLE4','PRICKLE3','INVS','DAAM1','DAAM2','RHOA','ROCK2','RAC1','RAC2','RAC3','MAPK8','MAPK10','MAPK9','PLCB1','PLCB2','PLCB3','PLCB4','CAMK2A','CAMK2D','CAMK2B','CAMK2G','PPP3CA','PPP3CB','PPP3CC','PPP3R1','PPP3R2','PRKCA','PRKCB','PRKCG','NFATC1','NFATC2','NFATC3','NFATC4']
neuroblastoma_genes=['RUNX1','CSF1R','MPO','CSF2','IL3','RUNX1T1','HDAC1','HDAC2','SIN3A','NCOR1','CEBPA','PER2','SPI1','CD14','ITGAM','FCGR1A','JUP','PML','RARA','CCNA2','CCNA1','CEBPE','BCL2A1','ZBTB16','MYC','DUSP6','TCF3','PBX1','WNT16','ETV6','ETV7','DEFA1','DEFA3','DEFA4','DEFA5','DEFA6','DEFA1B','ELANE','GZMB','KMT2A','AFF1','CDK9','CCNT1','CCNT2','MLLT1','MLLT3','DOT1L','LMO2','PBX3','RUNX2','SMAD1','KLF3','MEF2C','HOXA9','HOXA10','JMJD1C','HMGA2','KDM6A','SUPT3H','PROM1','FLT3','BMP2K','IGF1R','CDKN1B','CDK14','MEIS1','HOXA11','SIX1','SIX4','EYA1','CDKN2C','HPGD','GRIA3','FUT8','TLX3','TLX1','BCL11B','LDB1','LYL1','HHEX','PTCRA','REL','CCND2','BIRC2','BIRC3','TRAF1','BCL2L1','CD86','CD40','BCL6','IGH','MAF','ITGB7','NSD2','H3-5','H3-3B','H3C4','H3C3','H3C1','H3-3A','H3-4','H3C14','H3C15','H3C13','H3C6','H3C11','H3C8','H3C12','H3C10','H3C2','H3C7','PAX5','PAX8','PPARG','RXRA','RXRB','RXRG','PRCC','TFE3','CDKN1A','TMPRSS2','ERG','PLAU','PLAT','MMP3','MMP9','ZEB1','IL1R2','SPINT1','ETV1','ETV4','ETV5','SLC45A3','ELK4','DDX5','MYCN','MAX','MDM2','PTK2','TP53','BMI1','COMMD3-BMI1','SP1','ZBTB17','NTRK1','NGFR','MEN1','EWSR1','FLI1','IGF1','ID2','TGFBR2','IGFBP3','FEV','ATF1','ARNT2','ATM','MITF','WT1','PDGFA','IL2RB','BAIAP3','TSPAN7','MLF1','NR4A3','TAF15','FUS','DDIT3','CEBPB','IL6','NFKBIZ','NFKB1','RELA','CXCL8','PAX7','PAX3','FOXO1','FLT1','SS18','SSX1','SSX2','SSX2B','NUPR1','ASPSCR1','MET','GADD45A','GADD45B','GADD45G','BAX','BAK1','DDB2','POLK']
cancer_genes = medulloblastoma_genes
medulloblastoma_cell_lines = ['D-283MED','Daoy','ONS-76','PFSK-1']
omics_data_path = '/home/tol/paccmann_generator/data/gdsc_transcriptomics_for_conditional_generation.pkl'
protein_data_path = '/mnt/c/Users/PatriciaStoll/Documents/data/embedding.csv'#paccmann_affinity/sars_cov2_data/tape/transformer/avg.csv'
protein_data_seq_path = '/home/tol/paccmann_generator/output.csv'

# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('train_paccmann_rl')
logger_m = logging.getLogger('matplotlib')
logger_m.setLevel(logging.WARNING)

disable_rdkit_logging()

params = dict()
params['site'] = site
params['cancertype'] = cancertype


#logger.info(f'Model with name {model_name} starts.')

# Load omics profiles for conditional generation,
# complement with avg per site
omics_df = pd.read_pickle(omics_data_path)
omics_df = add_avg_profile(omics_df)
idx = [i in medulloblastoma_cell_lines for i in omics_df['cell_line']]
omics_df  = omics_df[idx]
print("omics data:", omics_df.shape, omics_df['cell_line'].iloc[0])
#test_cell_line = omics_df['cell_line'].iloc[0]
#model_name = model_name + '_' + test_cell_line
#omics_df = omics_df[omics_df.histology == cancertype]
print(omics_df.shape, omics_df['cell_line'])

# Load protein sequence data
#if protein_data_path.endswith('.smi'):
#    protein_df = read_smi(protein_data_path, names=['Sequence'])
#elif protein_data_path.endswith('.csv'):
#    protein_df = pd.read_csv(protein_data_path, index_col=0, header=None, names=[str(x) for x in range(768)]) #'entry_name')
#else:
#    raise TypeError(
#        f"{protein_data_path.split('.')[-1]} files are not supported."
#    )

protein_df = pd.read_csv(protein_data_path, index_col=0)#, header=None, names=[str(x) for x in range(768)]) #'entry_name')
protein_df = protein_df[~protein_df.index.isnull()]
protein_df.index = [i.split('|')[2] for i in protein_df.index]
protein_seq_df = pd.read_csv(protein_data_seq_path, names = ['sequence'], index_col=0) #, index_col='entry_name')
#print(protein_seq_df.head)
protein_seq_df.index = [i.split('|')[2] for i in protein_seq_df.index]
protein_df = pd.concat([protein_df, protein_seq_df], axis=1, join='outer')
protein_df = protein_df[[s.split('_')[0] in cancer_genes for s in protein_df.index]]
#print(protein_df.head)
print("proteins:", protein_df.shape, len(cancer_genes))
