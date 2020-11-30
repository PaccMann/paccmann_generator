from pytoda.preprocessing.combat import combat
import pandas as pd
from sklearn.preprocessing import StandardScaler
from paccmann_generator.utils import add_avg_profile

scaler = StandardScaler()
data2 = pd.read_pickle('data/gdsc_transcriptomics_for_conditional_generation.pkl')
data2 = add_avg_profile(data2)
df = pd.DataFrame(index=data2['cell_line'], columns=range(2128))
print(data2.head(3))
for cell in data2['cell_line']:
    df[cell] = data2[data2['cell_line']==cell]#['gene_expression'].values
print(df.head(3))
print(pd.DataFrame(data2['gene_expression'].values))
1/0
print(data2['gene_expression'].mean(axis = 0))
print(data2['gene_expression'].std(axis = 0))
data = pd.read_csv('data/XenograftNeuroblastomaSY5YEntrectinibResistance-Brodeur-18-fpkm-ensh37e75.txt', sep='\t', index_col=0)
data = data.drop(columns=['probeset'])
print(data.shape, data.T.head(2))
scaled_data = pd.DataFrame(scaler.fit_transform(data.T), index= data.columns, columns= data.index)
print(scaled_data.shape)
print(scaled_data.head(3))
print(scaled_data.mean(axis = 0))
print(scaled_data.std(axis = 0))