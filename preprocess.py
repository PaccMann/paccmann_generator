from pytoda.preprocessing.combat import combat
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from paccmann_generator.utils import add_avg_profile
from scipy.stats import wilcoxon
from scipy.spatial.distance import jensenshannon

scaler = StandardScaler()
data2 = pd.read_pickle('data/gdsc_transcriptomics_for_conditional_generation.pkl')
data2 = add_avg_profile(data2)
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
print(df.index)
print(scaled_data.columns)
cells = df.index
overlap = [cell for cell in cells if cell in scaled_data.columns]
print("overlapping on cell lines:", overlap)

# wilcoxon test:
print(wilcoxon(scaled_data, data2))

#jensen shanon:
print(jensenshannon(scaled_data, data2))