import pandas as pd
import numpy as np
from matplotlib import pylab as plt
import seaborn as sns


def plot_without_label(df, saving_path, cmap, vmax=1):
	sns.set_theme(style="white")
	mask = np.triu(np.ones_like(df, dtype=bool))
	f, ax = plt.subplots(figsize=(7, 7))
	# Draw the heatmap with the mask and correct aspect ratio
	sns.heatmap(df, vmin=0, vmax=vmax, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, xticklabels=False, yticklabels=False, cbar_kws={"shrink": .5})
	plt.tight_layout()
	plt.savefig(saving_path)
	# plt.show()



paths = {
			"Louvain_Pearson":"df_Louvain_Pearson.xlsx", 
			"Pearson":"pearson.xlsx", 
			}

for diag in ['CN', 'SMC', 'eMCI', 'lMCI']:
	print("doing ", diag)
	for key in paths.keys():
		print("plotting ", key)
		path_df_matrix = paths[key]
		full_path_df_matrix = diag + '/' + path_df_matrix
		df_matrix = pd.read_excel(full_path_df_matrix)
		df_matrix = df_matrix.set_index("Unnamed: 0")
		saving_path = diag + '/figures/' + path_df_matrix[:-4] + 'png'
		# Generate a custom diverging colormap
		cmap = sns.cubehelix_palette(start=0, rot=-.5, light=2.5, dark=0.4, as_cmap=True)
		plot_without_label(df_matrix, saving_path, cmap)
plt.close('all')



saving_path = 'CN/figures/df_louvain_Pearson_cleaned_age_tivpng'
df_matrix = pd.read_excel("CN/df_louvain_Pearson_cleaned_age_tiv.xlsx")
df_matrix = df_matrix.set_index("Unnamed: 0")
plot_without_label(df_matrix, saving_path, cmap)

