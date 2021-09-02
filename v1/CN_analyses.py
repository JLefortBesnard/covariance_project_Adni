# from IPython import get_ipython
# get_ipython().magic('reset -sf')

import all_functions
import numpy as np
import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler

np.random.seed(0)

# extract all file from folder
files = glob.glob('C:\\Users\\lefortb211\\Downloads\\ADNI_CN_smwc\\*')

# Sort and store nifti paths of gm, wm, and lcr into a dataframe
df = all_functions.dataframize_nii_paths(files)

# compute TIV and store it in the df
df = all_functions.compute_store_tiv(df)

# add age, group and sex information
df_demog = pd.read_csv("CN/CN_3T_6_23_2021.csv")
df_demog = df_demog[['Subject', 'Group', 'Sex', 'Age']] # subject, group, sex, age
df_demog = df_demog.set_index('Subject')
df = all_functions.merge_df_by_index(df, df_demog)
# add executive scores (file -containing each diag info- in CN folder to avoid duplicates)
df_executive = pd.read_excel("CN/score_ex.xlsx") 
df_executive = df_executive.set_index('PTID')
df = all_functions.merge_df_by_index(df, df_executive)

#################
## extract vbm ##
#################

# df_FS = all_functions.fit_atlas(df)
# df_FS.to_excel('CN/computed_data/df_fs.xlsx')



# reload it
df_FS = pd.read_excel("CN/computed_data/df_fs.xlsx").set_index('Unnamed: 0')
labels = list(df_FS.columns)
# standradize vbm and clean signal for tiv and age
# standardize values
df_FS_ss = pd.DataFrame(columns=df_FS.columns, index=df_FS.index, data=StandardScaler().fit_transform(df_FS.values))
info = df[['TIV', 'TIV_gm', 'TIV_wm', 'TIV_lcr', 'Group', 'Sex', 'Age', 'ADNI_MEM', 'ADNI_EF']]
df = all_functions.merge_df_by_index(info, df_FS_ss)
df_cleaned_AT = all_functions.clean_signal(df, labels, ['TIV', 'Age'])
df_cleaned_ATE = all_functions.clean_signal(df_cleaned_AT, labels, ['ADNI_EF'])



df_pearson_AT = all_functions.compute_pearson(df_cleaned_AT[labels])
df_pearson_AT.to_excel("CN/computed_data/df_pearson_AT.xlsx")
df_pearson_ATE = all_functions.compute_pearson(df_cleaned_ATE[labels])
df_pearson_ATE.to_excel("CN/computed_data/df_pearson_ATE.xlsx")
communities_AT = all_functions.compute_louvain_community(df_pearson_AT[labels])
np.save("CN/computed_data/communities_AT",communities_AT)
communities_ATE = all_functions.compute_louvain_community(df_pearson_ATE[labels])
np.save("CN/computed_data/communities_ATE",communities_ATE)
df_louvain_AT = all_functions.reorganize_with_louvain_community(df_pearson_AT[labels], communities_AT)
df_louvain_AT.to_excel("CN/computed_data/df_louvain_AT.xlsx")
df_louvain_ATE = all_functions.reorganize_with_louvain_community(df_pearson_ATE[labels], communities_ATE)
df_louvain_ATE.to_excel("CN/computed_data/df_louvain_ATE.xlsx")

all_functions.plot_matrice(df_pearson_AT, "pearson_AT", cmap=None, with_labels=False, saving_path="CN/figures/pearson_AT.png", show=0)
all_functions.plot_matrice(df_pearson_ATE, "pearson_ATE", cmap=None, with_labels=False, saving_path="CN/figures/pearson_ATE.png", show=0)
all_functions.plot_matrice(df_louvain_AT, "louvain_AT", cmap=None, with_labels=False, saving_path="CN/figures/louvain_AT.png", show=0)
all_functions.plot_matrice(df_louvain_ATE, "louvain_ATE", cmap=None, with_labels=False, saving_path="CN/figures/louvain_ATE.png", show=0)

df_mat_diff = all_functions.matrix_difference(df_cleaned_AT[labels], df_cleaned_ATE[labels], n_perm=1000)
df_mat_diff.to_excel("CN/computed_data/df_differences.xlsx")

