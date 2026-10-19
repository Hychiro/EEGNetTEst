import pandas as pd
import numpy as np
import os
from bciflow.modules.analysis.metric_functions import accuracy


# accDict = {
#     # 'Eegnet1' : [],
#     # 'Eegnet2' : [],
#     'Eegnet3' : [],
#     'Eegnet4' : [],
#     'Eegnet5' : [],
#     'Eegnet6' : [],
#     'Eegnet7' : [],
#     'Eegnet8' : [],
#     }

accDict = {'2a': [],
        '2b': [],
        'cbcic': []
        }
        
# auxFold = "EEGResults"


auxFold = {'2a': "Results/2a",
        '2b': "Results/2b",
        'cbcic': "Results/cbcic"
        }
for key in accDict:
    T = 11 if key == "cbcic" else 10
    for i in range(1,T):
        print(f"Processing {key} subject {i}...")
        for filename in os.listdir(auxFold[key]):
            if filename.endswith(".csv") and filename.startswith(f"Eegnet6_subject_{i}_"):
                filepath = os.path.join(auxFold[key], filename)
                df = pd.read_csv(filepath)
                accDict[key].append(accuracy(df))


for key in accDict.keys():
    vals = np.array(accDict[key])
    mean = np.mean(vals)
    std = np.std(vals)
    accDict[key].append(mean)
    accDict[key].append(std)
print(accDict)

max_len = max(len(accDict['2a']), len(accDict['2b']), len(accDict['cbcic']))
for key in accDict.keys():
    while len(accDict[key]) < max_len:
        accDict[key].append(0)

final = pd.DataFrame(accDict)
final.to_csv("TabelaFinalEEGNetAllDatasets2.csv", index=False)
