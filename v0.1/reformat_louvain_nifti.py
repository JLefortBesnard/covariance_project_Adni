import pandas as pd
import numpy as np
from matplotlib import pylab as plt
import nibabel as nib
from nilearn import datasets as ds
import community
import networkx as nx


def save_comunity(df_cov, saving_path=None):
	''' Compute network graph, then louvain community
	and save as nifti image
	
	Parameters
	----------
	df_cov : pandas dataframe of the covariance matrix (n_roi*n_roi)

	saving_path : str
		if not None, saved at the given path as .nii

	'''
	# reload atlas
	atlas = ds.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm',  symmetric_split=True)
	labels = atlas.labels[1:]
	# compute the best partition
	G = nx.from_numpy_matrix(df_cov.values)  
	nx.draw(G, with_labels=True) 
	partition = community.best_partition(G, random_state=0)
	np.save(saving_path, partition)
	return partition


partitions = []
for diag in ['CN', 'SMC', 'eMCI', 'lMCI']:
	df_pearson = pd.read_excel("{}/pearson.xlsx".format(diag))
	df_pearson = df_pearson.set_index('Unnamed: 0')
	partition = save_comunity(df_pearson, saving_path="{}/pearson_partition".format(diag))
	partitions.append(partition)

partition_CN = partitions[0]
partition_SMC = partitions[1]
partition_eMCI = partitions[2]
partition_lMCI = partitions[3]

partition_CN = np.load("CN/pearson_partition.npy", allow_pickle=True).item()
partition_SMC = np.load("SMC/pearson_partition.npy", allow_pickle=True).item()
partition_eMCI = np.load("eMCI/pearson_partition.npy", allow_pickle=True).item()
partition_lMCI = np.load("lMCI/pearson_partition.npy", allow_pickle=True).item()


atlas = ds.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm',  symmetric_split=True)

for key in range(96):
	print(atlas.labels[key+1], " : ")
	print(partition_CN[key])
	print(partition_SMC[key])
	print(partition_lMCI[key])
	print(partition_eMCI[key])

def update_dic(dic, list_keys, new_value):
	for key in list_keys:
		dic[key] = new_value
	return dic

# reorganize the partition following the same order (frontal, occipital, temporal, insular, parietal and PCC)
# to allow for standardized compaizon of community

# standardize frontal network 
keys = np.where(np.fromiter(partition_CN.values(), dtype=int) == 2) # CN
partition_CN = update_dic(partition_CN, keys[0], 100)
keys = np.where(np.fromiter(partition_SMC.values(), dtype=int) == 0) # SMC
partition_SMC = update_dic(partition_SMC, keys[0], 100)
keys = np.where(np.fromiter(partition_eMCI.values(), dtype=int) == 2) # eMCI 
partition_eMCI = update_dic(partition_eMCI, keys[0], 100)
keys = np.where(np.fromiter(partition_lMCI.values(), dtype=int) == 0) #lMCI
partition_lMCI = update_dic(partition_lMCI, keys[0], 100)

# standardize Occipital network 
keys = np.where(np.fromiter(partition_CN.values(), dtype=int) == 0) # CN 
partition_CN = update_dic(partition_CN, keys[0], 101)
keys = np.where(np.fromiter(partition_SMC.values(), dtype=int) == 3) # SMC
partition_SMC = update_dic(partition_SMC, keys[0], 101)
keys = np.where(np.fromiter(partition_eMCI.values(), dtype=int) == 5) # eMCI 
partition_eMCI = update_dic(partition_eMCI, keys[0], 101)
keys = np.where(np.fromiter(partition_lMCI.values(), dtype=int) == 2) #lMCI
partition_lMCI = update_dic(partition_lMCI, keys[0], 101)

# standardize Temporal network 
keys = np.where(np.fromiter(partition_CN.values(), dtype=int) == 4) # CN 
partition_CN = update_dic(partition_CN, keys[0], 102)
keys = np.where(np.fromiter(partition_SMC.values(), dtype=int) == 2) # SMC
partition_SMC = update_dic(partition_SMC, keys[0], 102)
keys = np.where(np.fromiter(partition_eMCI.values(), dtype=int) == 3) # eMCI 
partition_eMCI = update_dic(partition_eMCI, keys[0], 102)
keys = np.where(np.fromiter(partition_lMCI.values(), dtype=int) == 3) #lMCI
partition_lMCI = update_dic(partition_lMCI, keys[0], 102)


# standardize Insula network 
keys = np.where(np.fromiter(partition_CN.values(), dtype=int) == 1) # CN 
partition_CN = update_dic(partition_CN, keys[0], 103)
keys = np.where(np.fromiter(partition_SMC.values(), dtype=int) == 1) # SMC
partition_SMC = update_dic(partition_SMC, keys[0], 103)
keys = np.where(np.fromiter(partition_eMCI.values(), dtype=int) == 0)  # eMCI
partition_eMCI = update_dic(partition_eMCI, keys[0], 103)
keys = np.where(np.fromiter(partition_lMCI.values(), dtype=int) == 1) #lMCI
partition_lMCI = update_dic(partition_lMCI, keys[0], 103)

# standardize Parietal network 
keys = np.where(np.fromiter(partition_CN.values(), dtype=int) == 3) # CN 
partition_CN = update_dic(partition_CN, keys[0], 104)
# NO PARIETAL COMMUNITY IN SMC, PART OF FRONTAL    					# SMC
keys = np.where(np.fromiter(partition_eMCI.values(), dtype=int) == 1)  # eMCI
partition_eMCI = update_dic(partition_eMCI, keys[0], 104)
# NO PARIETAL COMMUNITY IN lMCI, PART OF FRONTAL    				# lMCI

# standardize PCC network 
keys = np.where(np.fromiter(partition_eMCI.values(), dtype=int) == 4)  # eMCI
partition_eMCI = update_dic(partition_eMCI, keys[0], 105)



np.save("CN/partition_reorganized.npy", partition_CN)
np.save("SMC/partition_reorganized.npy", partition_SMC)
np.save("eMCI/partition_reorganized.npy", partition_eMCI)
np.save("lMCI/partition_reorganized.npy", partition_lMCI)



# reorganize the df according to partition
def louvainize(df_cov, partition):
	''' Compute network graph, then louvain community
	and reorganized it into a matrix, and plot it

	Parameters
	----------
	df_cov : pandas dataframe of the covariance matrix (n_roi*n_roi)

	partition : dictionnary (roi/partition)

	'''
	louvain = np.zeros(df_cov.values.shape).astype(df_cov.values.dtype)
	labels = df_cov.columns
	labels_new_order = []
	i = 0
	# iterate through all created community
	for values in np.unique(list(partition.values())):
		# iterate through each ROI
		for key in partition:
			if partition[key] == values:
				louvain[i] = df_cov.values[key]
				labels_new_order.append(labels[key])
				i += 1
	# check positionning from original matrix to louvain matrix
	# get index of first roi linked to community 0
	index_roi_com0_louvain = list(partition.values()).index(100)
	# get nb of roi in community 0 (used to double checking)
	nb_com0 = np.unique(list(partition.values()), return_counts=True)[1][0]
	# get index of first roi linked to community 1
	index_roi_com1_louvain = list(partition.values()).index(101)
	assert louvain[0].sum() == df_cov.values[index_roi_com0_louvain].sum()
	assert louvain[nb_com0].sum() == df_cov.values[index_roi_com1_louvain].sum() 

	df_louvain = pd.DataFrame(index=labels_new_order, columns=labels_new_order, data=louvain)
	return df_louvain


import pandas as pd
import numpy as np
from matplotlib import pylab as plt
import seaborn as sns

# plot matrices
def plot_without_label(df, saving_path, cmap, vmax=1):
	sns.set_theme(style="white")
	mask = np.triu(np.ones_like(df, dtype=bool))
	f, ax = plt.subplots(figsize=(7, 7))
	# Draw the heatmap with the mask and correct aspect ratio
	sns.heatmap(df, vmin=0, vmax=vmax, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, xticklabels=False, yticklabels=False, cbar_kws={"shrink": .5})
	ax.hlines([3, 6, 9], *ax.get_xlim())
	plt.tight_layout()
	plt.savefig(saving_path)
	# plt.show()

# plot the newly reorganized louvain community for standardized comparizon of communitie
for diag in ['CN', 'SMC', 'eMCI', 'lMCI']:
	df_pearson = pd.read_excel('{}/pearson.xlsx'.format(diag)).set_index('Unnamed: 0')
	partition = np.load('{}/partition_reorganized.npy'.format(diag), allow_pickle=True).item()
	df_louvain_pearson = louvainize(df_pearson, partition)
	saving_path_df = '{}/df_louvain_Pearson_reoganized.xlsx'.format(diag)
	df_louvain_pearson.to_excel(saving_path_df)
	title = '{}_Louvain_reoganized'.format(diag)
	saving_path_fig = '{}/figures/{}.png'.format(diag, title)
	# Generate a custom diverging colormap
	cmap = sns.cubehelix_palette(start=0, rot=-.5, light=2.5, dark=0.4, as_cmap=True)
	plot_without_label(df_louvain_pearson, saving_path_fig, cmap)
plt.close('all')



# plot the newly reorganized louvain community for standardized comparizon of communities with line for 
# delimitating the communities
diag = 'lMCI'
partition = np.load('{}/partition_reorganized.npy'.format(diag), allow_pickle=True).item()
np.unique(list(partition.values()), return_counts=True)

df_louvain_pearson = pd.read_excel('{}/df_louvain_Pearson_reoganized.xlsx'.format(diag)).set_index('Unnamed: 0')
sns.set_theme(style="white")
mask = np.triu(np.ones_like(df_louvain_pearson, dtype=bool))
f, ax = plt.subplots(figsize=(7, 7))
	# Draw the heatmap with the mask and correct aspect ratio
cmap = sns.cubehelix_palette(start=0, rot=-.5, light=2.5, dark=0.4, as_cmap=True)
sns.heatmap(df_louvain_pearson, vmin=0, vmax=1, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, xticklabels=False, yticklabels=False, cbar_kws={"shrink": .5})
ax.hlines([27], *ax.get_xlim(), color='green', label='Occipital')
ax.vlines([27], *ax.get_ylim(), color='green', label='Occipital')
ax.hlines([54], *ax.get_xlim(), color='blue', label='Temporal')
ax.vlines([54], *ax.get_ylim(), color='blue', label='Temporal')
ax.hlines([81], *ax.get_xlim(), color='red', label='Insula')
ax.vlines([81], *ax.get_ylim(), color='red', label='Insula')
# ax.hlines([75], *ax.get_xlim(), color='pink', label='Parietal')
# ax.vlines([75], *ax.get_ylim(), color='pink', label='Parietal')
# ax.hlines([88], *ax.get_xlim(), color='pink', label='PCC')
# ax.vlines([88], *ax.get_ylim(), color='pink', label='PCC')
plt.legend()
plt.tight_layout()
plt.show()



# reorganised ACCORDING to CN partition
for diag in ['CN', 'SMC', 'eMCI', 'lMCI']:
	df_pearson = pd.read_excel('{}/pearson.xlsx'.format(diag)).set_index('Unnamed: 0')
	partition = np.load('CN/partition_reorganized.npy', allow_pickle=True).item()
	df_louvain_pearson = louvainize(df_pearson, partition)
	title = '{}_Louvain_reoganized_as_CN'.format(diag)
	saving_path_fig = '{}/figures/{}.png'.format(diag, title)
	# Generate a custom diverging colormap
	cmap = sns.cubehelix_palette(start=0, rot=-.5, light=2.5, dark=0.4, as_cmap=True)
	plot_without_label(df_louvain_pearson, saving_path_fig, cmap)
plt.close('all')