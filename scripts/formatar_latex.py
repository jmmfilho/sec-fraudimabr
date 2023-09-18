import os
import pandas as pd
from math import floor

data_dir = '../results_telegram/'
linha = ''

for filename in os.listdir(data_dir):
    linha = ''
    df = pd.read_csv(data_dir+filename)

    linha += f"""LR & {floor(df.iloc[0]['roc_auc_avg']*100)/100}\pm{round(df.iloc[0]['roc_auc_std'],2)} & {floor(df.iloc[0]['precision_avg']*100)/100}\pm{round(df.iloc[0]['precision_std'],2)} & {floor(df.iloc[0]['recall_avg']*100)/100}\pm{round(df.iloc[0]['recall_std'],2)} & {floor(df.iloc[0]['f1_avg']*100)/100}\pm{round(df.iloc[0]['f1_std'],2)} \\\\
BNB & {floor(df.iloc[1]['roc_auc_avg']*100)/100}\pm{round(df.iloc[1]['roc_auc_std'],2)} & {floor(df.iloc[1]['precision_avg']*100)/100}\pm{round(df.iloc[1]['precision_std'],2)} & {floor(df.iloc[1]['recall_avg']*100)/100}\pm{round(df.iloc[1]['recall_std'],2)} & {floor(df.iloc[1]['f1_avg']*100)/100}\pm{round(df.iloc[1]['f1_std'],2)} \\\\
MNB & {floor(df.iloc[2]['roc_auc_avg']*100)/100}\pm{round(df.iloc[2]['roc_auc_std'],2)} & {floor(df.iloc[2]['precision_avg']*100)/100}\pm{round(df.iloc[2]['precision_std'],2)} & {floor(df.iloc[2]['recall_avg']*100)/100}\pm{round(df.iloc[2]['recall_std'],2)} & {floor(df.iloc[2]['f1_avg']*100)/100}\pm{round(df.iloc[2]['f1_std'],2)} \\\\
LSVC & {floor(df.iloc[3]['roc_auc_avg']*100)/100}\pm{round(df.iloc[3]['roc_auc_std'],2)} & {floor(df.iloc[3]['precision_avg']*100)/100}\pm{round(df.iloc[3]['precision_std'],2)} & {floor(df.iloc[3]['recall_avg']*100)/100}\pm{round(df.iloc[3]['recall_std'],2)} & {floor(df.iloc[3]['f1_avg']*100)/100}\pm{round(df.iloc[3]['f1_std'],2)} \\\\
KNN & {floor(df.iloc[4]['roc_auc_avg']*100)/100}\pm{round(df.iloc[4]['roc_auc_std'],2)} & {floor(df.iloc[4]['precision_avg']*100)/100}\pm{round(df.iloc[4]['precision_std'],2)} & {floor(df.iloc[4]['recall_avg']*100)/100}\pm{round(df.iloc[4]['recall_std'],2)} & {floor(df.iloc[4]['f1_avg']*100)/100}\pm{round(df.iloc[4]['f1_std'],2)} \\\\
SGD & {floor(df.iloc[5]['roc_auc_avg']*100)/100}\pm{round(df.iloc[5]['roc_auc_std'],2)} & {floor(df.iloc[5]['precision_avg']*100)/100}\pm{round(df.iloc[5]['precision_std'],2)} & {floor(df.iloc[5]['recall_avg']*100)/100}\pm{round(df.iloc[5]['recall_std'],2)} & {floor(df.iloc[5]['f1_avg']*100)/100}\pm{round(df.iloc[5]['f1_std'],2)} \\\\
RF & {floor(df.iloc[6]['roc_auc_avg']*100)/100}\pm{round(df.iloc[6]['roc_auc_std'],2)} & {floor(df.iloc[6]['precision_avg']*100)/100}\pm{round(df.iloc[6]['precision_std'],2)} & {floor(df.iloc[6]['recall_avg']*100)/100}\pm{round(df.iloc[6]['recall_std'],2)} & {floor(df.iloc[6]['f1_avg']*100)/100}\pm{round(df.iloc[6]['f1_std'],2)} \\\\
GB & {floor(df.iloc[7]['roc_auc_avg']*100)/100}\pm{round(df.iloc[7]['roc_auc_std'],2)} & {floor(df.iloc[7]['precision_avg']*100)/100}\pm{round(df.iloc[7]['precision_std'],2)} & {floor(df.iloc[7]['recall_avg']*100)/100}\pm{round(df.iloc[7]['recall_std'],2)} & {floor(df.iloc[7]['f1_avg']*100)/100}\pm{round(df.iloc[7]['f1_std'],2)} \\\\
LR & {floor(df.iloc[8]['roc_auc_avg']*100)/100}\pm{round(df.iloc[8]['roc_auc_std'],2)} & {floor(df.iloc[8]['precision_avg']*100)/100}\pm{round(df.iloc[8]['precision_std'],2)} & {floor(df.iloc[8]['recall_avg']*100)/100}\pm{round(df.iloc[8]['recall_std'],2)} & {floor(df.iloc[8]['f1_avg']*100)/100}\pm{round(df.iloc[8]['f1_std'],2)} \\\\"""

    caminho_arquivo = f'../latex_telegram/{filename[:filename.find(".")]}'
    with open(caminho_arquivo,'w') as arq:
        arq.write(linha)
print(linha)
