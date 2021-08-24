import pandas as pd
import numpy as np
from matplotlib import pylab as plt
import seaborn as sns


def plot_without_label(df, saving_path, cmap):
	sns.set_theme(style="white")
	mask = np.triu(np.ones_like(df, dtype=bool))
	f, ax = plt.subplots(figsize=(7, 7))
	# Draw the heatmap with the mask and correct aspect ratio
	sns.heatmap(df, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, xticklabels=False, yticklabels=False, cbar_kws={"shrink": .5})
	plt.tight_layout()
	plt.savefig(saving_path)
	# plt.show()



paths = {
			"Louvain_cov":"df_Louvain_cov.xlsx", 
			"Louvain_LedoitWolf":"df_Louvain_LedoitWolf.xlsx", 
			"Louvain_LedoitWolfprec":"df_Louvain_LedoitWolf_prec.xlsx", 
			"Louvain_Pearson":"df_Louvain_Pearson.xlsx", 
			"LedoitWolf":"ledoiwolf_cov.xlsx",
			"LedoitWolfprec":"ledoiwolf_prec.xlsx", 
			"Pearson":"pearson.xlsx", 
			"Covariance":"cov.xlsx"
			}

for diag in ['CN', 'SMC', 'eMCI', 'lMCI']:
	for key in paths.keys():
		path_df_matrix = paths[key]
		full_path_df_matrix = diag + '/' + path_df_matrix
		df = 'CN/ledoiwolf_prec.xlsx'
		df_matrix = pd.read_excel(full_path_df_matrix)
		df_matrix = df_matrix.set_index("Unnamed: 0")
		print(key)
		print(np.tril(df_matrix, 1).max())
		saving_path = diag + '/figures/' + path_df_matrix[:-4] + 'png'
		# Generate a custom diverging colormap
		cmap = sns.color_palette("viridis", as_cmap=True)
		if 'prec' in saving_path:
			cmap = sns.light_palette("seagreen", as_cmap=True)
		plot_without_label(df_matrix, saving_path, cmap)
plt.close('all')





