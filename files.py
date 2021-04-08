#language_path = '/mnt/c/Users/PatriciaStoll/paccmann_affinity/trained_models/vae_selfies_one_hot_mod'
mol_model_path = '/mnt/c/Users/PatriciaStoll/paccmann_affinity/trained_models/vae_selfies_one_hot'
omics_model_path = '/home/tol/paccmann_generator/pvae'
protein_model_path = '/mnt/c/Users/PatriciaStoll/paccmann_affinity/trained_models/pevae_avg'
ic50_model_path = '/home/tol/paccmann_003'
affinity_model_path = '/mnt/c/Users/PatriciaStoll/paccmann_affinity/trained_models/base_affinity'
omics_data_path = '/home/tol/paccmann_generator/data/gdsc_transcriptomics_for_conditional_generation.pkl'
protein_data_path = '/mnt/c/Users/PatriciaStoll/Documents/data/embedding.csv'#paccmann_affinity/sars_cov2_data/tape/transformer/avg.csv'
protein_data_seq_path = '/home/tol/paccmann_generator/data/protein_data.csv'
set_encoder_path = '/home/tol/data/sets/results_njadata'
unbiased_predictions_path = '/home/tol/scripts/biased_models/unbiased_liver/results/generated.csv' #'/home/tol/paccmann_generator/biased_models/unbiased/results/generated.csv'
#protein_data_seq_path = '/mnt/c/Users/PatriciaStoll/paccmann_affinity/sars_cov2_data/uniprot_sars_cov2.csv'
params_path = '/home/tol/paccmann_generator/examples/IC50/example_params.json'
remove_invalid = True
model_name = 'sanitized'
site = 'liver'
cancertype = 'hepatoblastoma'
cancer_genes = ['PORCN','WNT1','WNT2','WNT2B','WNT3','WNT3A','WNT4','WNT5A','WNT5B','WNT6','WNT7A','WNT7B','WNT8A','WNT8B','WNT9A','WNT9B','WNT10B','WNT10A','WNT11','WNT16','CER1','NOTUM','WIF1','SERPINF1','SOST','DKK1','DKK2','DKK4','SFRP1','SFRP2','SFRP4','SFRP5','RSPO1','RSPO2','RSPO3','RSPO4','LGR4','LGR5','LGR6','RNF43','ZNRF3','FZD1','FZD7','FZD2','FZD3','FZD4','FZD5','FZD8','FZD6','FZD10','FZD9','LRP5','LRP6','BAMBI','CSNK1E','TPTEP2-CSNK1E','DVL3','DVL2','DVL1','FRAT1','FRAT2','CSNK2A1','CSNK2A2','CSNK2A3','CSNK2B','NKD1','NKD2','CXXC4','SENP2','GSK3B','CTNNB1','AXIN1','AXIN2','APC','APC2','CSNK1A1L','CSNK1A1','TCF7','TCF7L1','TCF7L2','LEF1','CTNNBIP1','CBY1','CHD8','SOX17','CTBP1','CTBP2','CTNND2','CREBBP','EP300','RUVBL1','SMAD4','SMAD3','MAP3K7','NLK','MYC','JUN','FOSL1','CCND1','CCND2','CCND3','CCN4','PPARD','MMP7','PSEN1','PRKACA','PRKACB','PRKACG','TP53','SIAH1','CACYBP','SKP1','TBL1X','TBL1Y','TBL1XR1','BTRC','FBXW11','CUL1','RBX1','GPC4','ROR1','ROR2','RYK','VANGL2','VANGL1','PRICKLE1','PRICKLE2','PRICKLE4','PRICKLE3','INVS','DAAM1','DAAM2','RHOA','ROCK2','RAC1','RAC2','RAC3','MAPK8','MAPK10','MAPK9','PLCB1','PLCB2','PLCB3','PLCB4','CAMK2A','CAMK2D','CAMK2B','CAMK2G','PPP3CA','PPP3CB','PPP3CC','PPP3R1','PPP3R2','PRKCA','PRKCB','PRKCG','NFATC1','NFATC2','NFATC3','NFATC4']
neuroblastoma_genes=['RUNX1','CSF1R','MPO','CSF2','IL3','RUNX1T1','HDAC1','HDAC2','SIN3A','NCOR1','CEBPA','PER2','SPI1','CD14','ITGAM','FCGR1A','JUP','PML','RARA','CCNA2','CCNA1','CEBPE','BCL2A1','ZBTB16','MYC','DUSP6','TCF3','PBX1','WNT16','ETV6','ETV7','DEFA1','DEFA3','DEFA4','DEFA5','DEFA6','DEFA1B','ELANE','GZMB','KMT2A','AFF1','CDK9','CCNT1','CCNT2','MLLT1','MLLT3','DOT1L','LMO2','PBX3','RUNX2','SMAD1','KLF3','MEF2C','HOXA9','HOXA10','JMJD1C','HMGA2','KDM6A','SUPT3H','PROM1','FLT3','BMP2K','IGF1R','CDKN1B','CDK14','MEIS1','HOXA11','SIX1','SIX4','EYA1','CDKN2C','HPGD','GRIA3','FUT8','TLX3','TLX1','BCL11B','LDB1','LYL1','HHEX','PTCRA','REL','CCND2','BIRC2','BIRC3','TRAF1','BCL2L1','CD86','CD40','BCL6','IGH','MAF','ITGB7','NSD2','H3-5','H3-3B','H3C4','H3C3','H3C1','H3-3A','H3-4','H3C14','H3C15','H3C13','H3C6','H3C11','H3C8','H3C12','H3C10','H3C2','H3C7','PAX5','PAX8','PPARG','RXRA','RXRB','RXRG','PRCC','TFE3','CDKN1A','TMPRSS2','ERG','PLAU','PLAT','MMP3','MMP9','ZEB1','IL1R2','SPINT1','ETV1','ETV4','ETV5','SLC45A3','ELK4','DDX5','MYCN','MAX','MDM2','PTK2','TP53','BMI1','COMMD3-BMI1','SP1','ZBTB17','NTRK1','NGFR','MEN1','EWSR1','FLI1','IGF1','ID2','TGFBR2','IGFBP3','FEV','ATF1','ARNT2','ATM','MITF','WT1','PDGFA','IL2RB','BAIAP3','TSPAN7','MLF1','NR4A3','TAF15','FUS','DDIT3','CEBPB','IL6','NFKBIZ','NFKB1','RELA','CXCL8','PAX7','PAX3','FOXO1','FLT1','SS18','SSX1','SSX2','SSX2B','NUPR1','ASPSCR1','MET','GADD45A','GADD45B','GADD45G','BAX','BAK1','DDB2','POLK']
