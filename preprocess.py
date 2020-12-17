from pytoda.preprocessing.combat import combat
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from paccmann_generator.utils import add_avg_profile
from scipy.stats import wilcoxon
from scipy.spatial.distance import jensenshannon
from matplotlib import pyplot as plt
import seaborn as sns
import sys
import json
import os
sys.path.append('/home/tol/interact')
from interact.uniprot import dict_from_uniprot
from loguru import logger
logger.disable("my_library")

mapping_name_path = 'data/gene_names_mapping.json' 
if (os.path.exists(mapping_name_path)):
    with open(mapping_name_path) as fp:
        gene_names = json.load(fp)
else:
    gene_names = dict_from_uniprot(key_name='HGNC')
    with open(mapping_name_path, 'w') as fp:
        json.dump(gene_names, fp)

hugo = pd.read_csv('data/hgnc-symbol-check.csv', header =1, index_col =0)
remove = hugo.index[hugo.index.duplicated()]
nans = hugo.index[hugo['HGNC ID'].isnull()]
hugo = hugo[~hugo.index.duplicated()]
hugo = hugo.drop(nans)#hugo.dropna(subset=['HGNC ID'])

logger.info(hugo.shape)
scaler = StandardScaler()
data2 = pd.read_pickle('data/gdsc_transcriptomics_for_conditional_generation.pkl')
gene_list = pd.read_pickle('data/2128_genes.pkl')
data2 = add_avg_profile(data2)
#print(data2.head(3))
df = pd.DataFrame(np.array([series.values for series in data2.gene_expression.values]), index=data2['cell_line'], columns=gene_list)
#print(df.head(3))
#print(df.mean(axis = 0)) # should be 0 
#print(df.std(axis = 0)) # should be 1

data = pd.read_csv('data/XenograftNeuroblastomaSY5YEntrectinibResistance-Brodeur-18-fpkm-ensh37e75.txt', sep='\t', index_col=0)
data = data.drop(columns=['probeset'])
#print(data.shape, data.T.head(2))
#transform data
transformed_data = np.arcsinh(data)
logger.info(transformed_data.shape)
logger.info(transformed_data.head(2))
#scale/normalize data
scaled_data = pd.DataFrame(scaler.fit_transform(data.T), index= data.columns, columns= data.index)
logger.info(scaled_data.shape)
logger.info(scaled_data.head(3))
logger.info(scaled_data.mean(axis = 0)) # should be 0
logger.info(scaled_data.std(axis = 0)) # should be 1

# how many cell lines overlap:
genes = set([str(i) for i in df.columns])
genes2 = set([i for i in scaled_data.columns if (i not in remove) or (i not in nans)])
logger.info(len(genes2))
liste2 = []
for gene in genes2:
    #print(gene, type(gene), (gene in hugo.index))
    if (gene in hugo.index) and (hugo.loc[gene,'HGNC ID'] in gene_names):
        liste2.append(gene_names[hugo.loc[gene,'HGNC ID']])

logger.info('overlap dict:', len(liste2), liste2[:10])
#sets
overlap = genes & genes2 #[gene for gene in genes if gene in genes2]
logger.info('overlapping on cell lines:', len(overlap))

# wilcoxon test:
#print(wilcoxon(scaled_data.values.flatten(), data2.values.flatten()))

def get_distributions(df, genes, bins=100):

    return pd.DataFrame(
        [np.histogram(df[gene].values, bins=bins, density=True)[0].tolist() for gene in genes],
        index=genes
    )

# jensen shanon:
# TO-DO: make historgrams to get probability distributions to use as input
scaled_data_prob = get_distributions(scaled_data, overlap, bins=4)
df_prob = get_distributions(df, overlap, bins=4)
js = []
for gene in overlap:
    logger.info(scaled_data_prob.loc[gene])
    logger.info(scaled_data_prob.loc[gene].shape) 
    logger.info(df_prob.loc[gene])
    logger.info(df_prob.loc[gene].shape)
    js.append(jensenshannon(scaled_data_prob.loc[gene], df_prob.loc[gene], 2))
#logger.enable("my_library")
logger.info(js)
sns.distplot(js, kde=True)
plt.xlim(0, 1)
plt.xlabel('jesens-shannon divergence')
plt.savefig('data/js.pdf')
#js = pd.DataFrame(pairwise_distances(scaled_data_prob, df_prob, metric=jensenshannon), index=overlap, columns=overlap)

#print(jensenshannon(scaled_data, data2))