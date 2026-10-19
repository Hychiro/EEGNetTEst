from bciflow.datasets.bciciv2a import bciciv2a
from bciflow.datasets.bciciv2b import bciciv2b
from bciflow.datasets.cbcic import cbcic
from bciflow.modules.core.kfold import kfold
from bciflow.modules.tf.bandpass import bandpass_conv, chebyshevII
from bciflow.modules.tf import cubic_resample,fft_resample
from bciflow.modules.analysis.metric_functions import accuracy
import torch
from eegnet import Eegnet
import numpy as np
import pandas as pd



# print(torch.__version__)
# print(torch.version.cuda)
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())



filterDict = {
    # 'Eegnet1' : [(bandpass_conv, {}),(Eegnet(),{})],
    # # 'Eegnet2' : [(chebyshevII,{}),(Eegnet(),{})],
    # 'Eegnet3' : [(cubic_resample,{'new_sfreq':128.0}),(Eegnet(),{})],
    # 'Eegnet4' : [(fft_resample,{'new_sfreq':128.0}),(Eegnet(),{})],
    'Eegnet5' : [(bandpass_conv, {}),(cubic_resample,{'new_sfreq':128.0}),(Eegnet(),{})],
    # 'Eegnet6' : [(bandpass_conv, {}),(fft_resample,{'new_sfreq':128.0}),(Eegnet(),{})],
    # 'Eegnet7' : [(chebyshevII, {}),(cubic_resample,{'new_sfreq':128.0}),(Eegnet(),{})],
    # 'Eegnet8' : [(chebyshevII, {}),(fft_resample,{'new_sfreq':128.0}),(Eegnet(),{})],
    }

# accDict = {
#     # 'Eegnet1' : [],
#     # 'Eegnet2' : [],
#     # 'Eegnet3' : [],
#     # 'Eegnet4' : [],
#     # 'Eegnet5' : [],
#     'Eegnet6' : [],
#     # 'Eegnet7' : [],
#     # 'Eegnet8' : [],
#     }

pathUsed1 = 'C:/Users/Hychiro/Documents/Ufjf/bci/testes no codigo do bciflow/Data/2a'
pathUsed2 = 'C:/Users/Hychiro/Documents/Ufjf/bci/testes no codigo do bciflow/Data/2b'
pathUsed3 = 'C:/Users/Hychiro/Documents/Ufjf/bci/testes no codigo do bciflow/Data/cbcic'
paths = [pathUsed1, pathUsed2, pathUsed3]
# se termina com o fim do path cria flag pra ele

for eachPath in paths:
    T = 11 if eachPath.endswith("cbcic") else 10 
    accDict = []
    for i in range(1,T):
        if eachPath.endswith("2a"):
            dataset = bciciv2a(subject=i, labels=["left-hand", "right-hand"], path=eachPath)
            print(f"labels: {dataset['y_dict']}, {np.unique(dataset['y'])}")
        elif eachPath.endswith("2b"):
            dataset = bciciv2b(subject=i, labels=["left-hand", "right-hand"], path=eachPath)
            print(f"labels: {dataset['y_dict']}, {np.unique(dataset['y'])}")
        elif eachPath.endswith("cbcic"):
            dataset = cbcic(subject=i, labels=["left-hand", "right-hand"], path=eachPath)
            print(f"labels: {dataset['y_dict']}, {np.unique(dataset['y'])}")
        for key in filterDict.keys():
            acc = 0
            print(f"labels: {dataset['y_dict']}, {np.unique(dataset['y'])}")
            print(f"Filter {key}")

            pre_folding = {}
            pos_folding = {}

            if len(filterDict[key]) == 2:
                pre_folding = {
                    'tf': filterDict[key][0],
                }
                pos_folding = {
                    'clf': filterDict[key][1]
                }
            elif len(filterDict[key]) == 3:
                pre_folding = {
                    'tf1': filterDict[key][0],
                    'tf2': filterDict[key][1],
                }
                pos_folding = {
                    'clf': filterDict[key][2]
                }
            
            ws =2.0
            results = kfold(
                target=dataset,
                start_window=dataset['events']['cue'][0] + 0.5,
                window_size=2.0,
                pre_folding=pre_folding,
                pos_folding=pos_folding
            )
            if eachPath.endswith("2a"):
                filename = f"Results/2a/{key}_subject_{i}_start_{dataset['events']['cue'][0] + 0.5:.2f}_window_{ws:.2f}".replace(".", "_") + ".csv"
            elif eachPath.endswith("2b"):
                filename = f"Results/2b/{key}_subject_{i}_start_{dataset['events']['cue'][0] + 0.5:.2f}_window_{ws:.2f}".replace(".", "_") + ".csv"
            elif eachPath.endswith("cbcic"):
                filename = f"Results/cbcic/{key}_subject_{i}_start_{dataset['events']['cue'][0] + 0.5:.2f}_window_{ws:.2f}".replace(".", "_") + ".csv"
            results.to_csv(filename, index=False,columns=['fold', 'tmin', 'true_label', *dataset["y_dict"].keys()])
            df = pd.DataFrame(results)
            acc = accuracy(df)
            accDict.append(np.copy(acc))
            print(f"{key} Accuracy: {acc:.4f}")

    df = pd.DataFrame(accDict)
    if eachPath.endswith("2a"):
        df.to_csv("acuracias2a.csv", index=False)
    elif eachPath.endswith("2b"):
        df.to_csv("acuracias2b.csv", index=False)
    elif eachPath.endswith("cbcic"):
        df.to_csv("acuraciascbcic.csv", index=False)

    # for key in filterDict.keys():
    #     vals = np.array(accDict[key])
    #     mean = np.mean(vals)
    #     std = np.std(vals)
    #     accDict[key].append(mean)
    #     accDict[key].append(std)

    # final = pd.DataFrame(accDict)
    # final.to_csv("TabelaFinalMBEEGNET.csv", index=False)
