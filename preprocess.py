from pytoda.preprocessing.combat import combat
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from paccmann_generator.utils import add_avg_profile
from scipy.stats import ranksums
from scipy.stats import wilcoxon
from scipy.stats import mannwhitneyu
from scipy.spatial.distance import jensenshannon
from scipy.stats import ttest_ind
from matplotlib import pyplot as plt
import seaborn as sns
import sys
import json
import os
sys.path.append('/home/tol/interact')
from interact.uniprot import dict_from_uniprot
from statsmodels.stats.multitest import multipletests
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
hugo = hugo.drop(nans) # hugo.dropna(subset=['HGNC ID'])

logger.info(hugo.shape)

scaler = StandardScaler()
scaler2 = StandardScaler()
hepatoblastoma_cell_lines = ['HUH-6-clone5','HuH-7','SNU-475','SNU-423','SNU-387','SNU-449','HLE','C3A']
medulloblastoma_cell_lines = ['D-283MED','Daoy','ONS-76','PFSK-1']
cancer_cell_lines = medulloblastoma_cell_lines
data2 = pd.read_pickle('data/gdsc_transcriptomics_for_conditional_generation.pkl')
gene_list = pd.read_pickle('data/2128_genes.pkl')
data2 = add_avg_profile(data2)
#idx = [i in cancer_cell_lines for i in data2['cell_line']]
#data2  = data2[idx]
#logger.info(data2.shape)
data2 = data2[(data2.site == 'central_nervous_system') | (data2.site == 'brain')]
#data2 = data2[data2.histology == 'neuroblastoma']
logger.info(data2.shape)
df = pd.DataFrame(np.array([series.values for series in data2.gene_expression.values]), index=data2['cell_line'], columns=gene_list)
#print(df.head(3))
print(df.mean(axis = 0)) # should be 0 
print(df.std(axis = 0)) # should be 1
# df = pd.DataFrame(scaler2.fit_transform(df), index= df.index, columns= df.columns)
# print(df.mean(axis = 0)) # should be 0 
# print(df.std(axis = 0)) # should be 1
# 1/0
files = 'TumorMedulloblastoma-Thompson-46-GCRMA-u133a'
data = pd.read_csv(f'data/{files}.txt', sep='\t', index_col=0)
data = data.drop(columns=['probeset']).astype(float)
#print(data.shape, data.T.head(2))
#transform data
print(data.head(3))
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
#genes2 = set([i for i in scaled_data.columns if (i not in remove) or (i not in nans)])
genes2 = set(str(i) for i in scaled_data.columns)
logger.info(len(genes2))
liste2 = []
for gene in genes2:
    #print(gene, type(gene), (gene in hugo.index))
    if (gene in hugo.index) and (hugo.loc[gene,'HGNC ID'] in gene_names):
        liste2.append(gene_names[hugo.loc[gene,'HGNC ID']])

logger.info('overlap dict:' + str(len(liste2)))
#sets
overlap = genes & genes2 #[gene for gene in genes if gene in genes2]
logger.info('overlapping on cell lines:' + str(len(overlap)))


def get_distributions(df, genes, bins=100):

    return pd.DataFrame(
        [np.histogram(df[gene].values, bins=bins, density=True)[0].tolist() for gene in genes],
        index=genes
    )

# jensen shanon:
bins=4
scaled_data_prob = get_distributions(scaled_data, overlap, bins=bins)
df_prob = get_distributions(df, overlap, bins=bins)
js = []
for gene in overlap:
    js.append(jensenshannon(scaled_data_prob.loc[gene], df_prob.loc[gene], 2))
#logger.enable("my_library")
logger.info(len(js))
sns.distplot(js, kde=True)
plt.xlim(0, 1)
plt.xlabel('jesen-shannon divergence')
plt.show()
plt.savefig(f'data/js_{files}.pdf')

# wilcoxon test:
#print(wilcoxon(scaled_data.values.flatten(), data2.values.flatten()))
pvals = []
#print(df.shape, scaled_data.shape)
for gene in overlap:
    sth = ranksums(scaled_data[gene], df[gene])
    pvals.append(ranksums(scaled_data[gene], df[gene])[1])
q_vals = multipletests(pvals, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)
# print(len(q_vals[1]), len(-np.log10(q_vals[1])))
# f, ax = plt.subplots(figsize=(7, 7))
# ax.set(xscale="log")
# #plt.xscale('log')
# #sns.distplot(q_vals[1], kde=True, ax=ax, bins=[0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]) # ax=ax
# #plt.xlim(0, 1)
# pd.DataFrame(q_vals[1]).plot.kde(ax=ax, legend=False)
# pd.DataFrame(q_vals[1]).plot.hist(density=True, ax=ax)
# plt.xlabel('q values')
# plt.show()
mask1 = q_vals[1] <= 0.001
mask2 = (q_vals[1] > 0.001) & (q_vals[1] <= 0.005)
mask3 = (q_vals[1] > 0.005) & (q_vals[1] <= 0.01)
mask4 = (q_vals[1] > 0.01) & (q_vals[1] <= 0.05)
mask5 = (q_vals[1] > 0.05) & (q_vals[1] <= 0.1)
mask6 = q_vals[1] > 0.1
plt.clf()
print('total', len(q_vals[1]), 'check', np.sum(mask1)+np.sum(mask2)+
np.sum(mask3)+np.sum(mask4)+np.sum(mask5)+np.sum(mask6))
plt.bar(1, np.sum(mask1)*100/len(q_vals[1]), color = 'red', label='<=0.001')
plt.bar(2, np.sum(mask2)*100/len(q_vals[1]), color = 'orange', label= '>0.001 & <=0.005')
plt.bar(3, np.sum(mask3)*100/len(q_vals[1]), color = 'yellowgreen', label= '>0.005 & <=0.01')
plt.bar(4, np.sum(mask4)*100/len(q_vals[1]), color = 'green', label= '>0.01 & <=0.05')
plt.bar(5, np.sum(mask5)*100/len(q_vals[1]), color = 'blue', label= '>0.05 & <=0.1')
plt.bar(6, np.sum(mask6)*100/len(q_vals[1]), color = 'violet', label= '>0.1')
plt.legend()
plt.title('percentage of q values')
plt.ylabel('%')
plt.show()
plt.savefig(f'data/wilcoxon_{files}.pdf')
sys.stdout =  open(f'data/statistical_tests_{files}.txt', 'w')
print("is <=0.001 significantlly different of the rest:"
    , ranksums(q_vals[1][mask1], q_vals[1][mask2 | mask3 | mask4 | mask5 | mask6])[1])
print("is <=0.005 significantlly different of the rest:"
    , ranksums(q_vals[1][mask2 | mask1], q_vals[1][mask3 | mask4 | mask5 | mask6])[1])
print("is <=0.01 significantlly different of the rest:"
    , ranksums(q_vals[1][mask2 | mask3 | mask1], q_vals[1][mask4 | mask5 | mask6])[1])
print("is <=0.05 significantlly different of the rest:"
    , ranksums(q_vals[1][mask2 | mask3 | mask4 | mask1], q_vals[1][mask5 | mask6])[1])
print("is <=0.1 significantlly different of the rest:"
    , ranksums(q_vals[1][mask2 | mask3 | mask4 | mask5 | mask1], q_vals[1][mask6])[1])
print("wilcoxon per gene:", q_vals[1])
q_vals_ranksum_flatten = ranksums(df[overlap].values.flatten(), scaled_data[overlap].values.flatten())
print("wilcoxon ranksums flatten:", q_vals_ranksum_flatten)
q_vals_mannwhitneyu_flatten = mannwhitneyu(df[overlap].values.flatten(), scaled_data[overlap].values.flatten())
print("wilcoxon mannwhitneyu flatten:", q_vals_mannwhitneyu_flatten)
plt.clf()
sys.stdout.close()

#js = pd.DataFrame(pairwise_distances(scaled_data_prob, df_prob, metric=jensenshannon), index=overlap, columns=overlap)

#print(jensenshannon(scaled_data, data2))