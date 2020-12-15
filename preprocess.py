from pytoda.preprocessing.combat import combat
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from paccmann_generator.utils import add_avg_profile
from scipy.stats import wilcoxon
from scipy.spatial.distance import jensenshannon
import sys
sys.path.append('/home/tol/interact')
from interact.uniprot import dict_from_uniprot

gene_names = dict_from_uniprot(key_name='HGNC')
hugo = pd.read_csv('data/hgnc-symbol-check.csv', header =1, index_col =0)
remove = hugo.index[hugo.index.duplicated()]
nans = hugo.index[hugo['HGNC ID'].isnull()]
hugo = hugo[~hugo.index.duplicated()]
hugo = hugo.drop(nans)#hugo.dropna(subset=['HGNC ID'])

print(hugo.shape)
scaler = StandardScaler()
data2 = pd.read_pickle('data/gdsc_transcriptomics_for_conditional_generation.pkl')
data2 = add_avg_profile(data2)
print(data2.head(3))
df = pd.DataFrame(index=data2['cell_line'], columns=range(2128))
for cell in data2['cell_line']:
    df.loc[cell,:] = data2[data2['cell_line']==cell]['gene_expression'].values[0].astype(float)
print(df.head(3))
print(df.mean(axis = 0)) # should be 0 
print(df.std(axis = 0)) # should be 1

data = pd.read_csv('data/XenograftNeuroblastomaSY5YEntrectinibResistance-Brodeur-18-fpkm-ensh37e75.txt', sep='\t', index_col=0)
data = data.drop(columns=['probeset'])
print(data.shape, data.T.head(2))
#transform data
transformed_data = np.arcsinh(data)
print(transformed_data.shape, transformed_data.head(2))
#scale/normalize data
scaled_data = pd.DataFrame(scaler.fit_transform(data.T), index= data.columns, columns= data.index)
print(scaled_data.shape)
print(scaled_data.head(3))
print(scaled_data.mean(axis = 0)) # should be 0
print(scaled_data.std(axis = 0)) # should be 1

# how many cell lines overlap:
genes = [str(i) for i in df.index]
genes2 = [i for i in scaled_data.columns if (i not in remove) or (i not in nans)]
print(len(genes2))
liste2 = []
for gene in genes2:
    #print(gene, type(gene), (gene in hugo.index))
    if (gene in hugo.index) and (hugo.loc[gene,'HGNC ID'] in gene_names):
        liste2.append(gene_names[hugo.loc[gene,'HGNC ID']])

print('overlap dict:', len(liste2))
overlap = [gene for gene in genes if gene in liste2]
print('overlapping on cell lines:', overlap)

# wilcoxon test:
print(wilcoxon(scaled_data, data2))

# jensen shanon:
# TO-DO: make historgrams to get probability distributions to use as input
print(jensenshannon(scaled_data, data2))