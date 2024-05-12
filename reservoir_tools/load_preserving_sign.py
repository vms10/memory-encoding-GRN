import networkx as nx
import pandas as pd

def load_preserving_signs(network):
    f=open(network)
    data=[]
    for line in f:
        e=line.strip()
        col=line.split()
        #print(len(col))
        if len(col)!=0:
            data.append(col)	
    source = []
    target = []
    weight = []
    for i in data:
        source.append(i[0])
        target.append(i[1])
        weight.append(i[2]) 
    weight_float = []
    for i in weight:
        if i == "+":
            weight_float.append(1.0)
        else:
            weight_float.append(-1.0)
    df= pd.DataFrame({'source': source,'target': target,'weight': weight_float})
    G=nx.from_pandas_edgelist(df, 'source', 'target', edge_attr = 'weight', create_using=nx.DiGraph)
    return G