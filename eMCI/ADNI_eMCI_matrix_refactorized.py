import nibabel as nib
import glob
import pandas as pd
import numpy as np
from nilearn import datasets as ds
from nilearn.image import resample_img
from nilearn.input_data import NiftiLabelsMasker
import psutil
from matplotlib import pylab as plt
import time
from nilearn.signal import clean
from nilearn import plotting, image
from sklearn.covariance import LedoitWolf
import seaborn as sns
from statsmodels.formula.api import ols
from sklearn.preprocessing import StandardScaler
import community
import networkx as nx

np.random.seed(0)

def dataframize_nii_paths(files):
	''' Sort and store nifti paths of gm, wm, and lcr into a dataframe

	Parameters
	----------
	files : list of path to nifti images
	
	'''
	# Sort cortex, white matter and LCR file
	files_per_sub = {}
	for file in files:
		split_name = file.split('_')
		sub_nb = split_name[3] + '_' + split_name[4] + '_' + split_name[5]
		assert '_S_' in sub_nb, "Problem with splitting path"
		for file in files:
			if sub_nb in file:
				if 'smwc1' in file:
					smwc1 = file
				elif 'smwc2' in file:
					smwc2 = file
				elif 'smwc3' in file:
					smwc3 = file
				else:
					print('pb with ', file)
					Stop
		files_per_sub[sub_nb] = [smwc1, smwc2, smwc3]
	# store it in a dataframe
	df = pd.DataFrame.from_dict(files_per_sub, orient='index')
	df.rename(columns={
            0: 'greyMatter_path',
            1: 'whiteMatter_path',
            2: 'LCR_path'}, inplace=True)
	# delete duplicates
	nb_duplicates = (df.index.duplicated() == True).sum()
	df = df[df.index.duplicated() == False]
	print('{} duplicates were removed. Df shape = {}'.format(nb_duplicates, df.shape))
	return df


def merge_df_by_index(df, df_2, method='outer'):
	''' merge 2 dataframe together on index, while making sure that there are no duplicates.
	
	Parameters
	----------
	df : pandas dataframe 
	df_2 : pandas dataframe
	
	'''
	df_merged = df.join(df_2, how=method)
	# delete nan if exists
	len1 = df_merged.__len__()
	df_merged = df_merged.dropna(axis=0, how='any')
	len2 = df_merged.__len__()
	print('{} nan were removed'.format(len1 - len2))
	# delete duplicates
	nb_duplicates = (df_merged.index.duplicated() == True).sum()
	df_merged = df_merged[df_merged.index.duplicated() == False]
	print('{} duplicates were removed. Df shape = {}'.format(nb_duplicates, df_merged.shape))

	return df_merged


def compute_store_tiv(df):
	''' compute TIV and store it in the df

	Parameters
	----------
	df : dataframe including the path of gm, wm and lcr per subject
	
	'''
	# create new columns to store the tiv
	df['TIV'] = '*'
	df['TIV_gm'] = '*'
	df['TIV_wm'] = '*'
	df['TIV_lcr'] = '*'

	# store the voxel size for future checking
	dim_expected = nib.load(df['greyMatter_path'].iloc[0]).header.values()[15]

	# iterate across subject to extract, compute and store the TIV
	for ind, row in enumerate(df.index):
		# load nifti images
		smwc1 = nib.load(df['greyMatter_path'].loc[row])
		smwc2 = nib.load(df['whiteMatter_path'].loc[row])
		smwc3 = nib.load(df['LCR_path'].loc[row])
		# check voxel dimension 
		assert smwc1.header.values()[15].sum() == dim_expected.sum()
		assert smwc2.header.values()[15].sum() == dim_expected.sum()
		assert smwc3.header.values()[15].sum() == dim_expected.sum()
		# compute TIV 
		tiv1 = smwc1.get_data().sum()/1000 # grey matter
		tiv2 = smwc2.get_data().sum()/1000 # white matter
		tiv3 = smwc3.get_data().sum()/1000 # LCR
		TIV = tiv1 + tiv2 + tiv3 # total
		assert TIV != 0, "Problem with participant {}".format(row) 
		assert tiv1 != 0, "Problem with participant {}".format(row)
		assert tiv2 != 0, "Problem with participant {}".format(row)
		assert tiv3 != 0, "Problem with participant {}".format(row)
		# online checking
		print(ind, ' / ', len(df.index))
		print("Sub ", row, " TIV = ", TIV)
		# store TIV in df
		df['TIV_gm'].loc[row] = tiv1
		df['TIV_wm'].loc[row] = tiv2
		df['TIV_lcr'].loc[row] = tiv3
		df['TIV'].loc[row] = TIV
	assert '*' not in df.values, "A tiv value seems to be missing"
	return df

def list_outliers(df, percentile=5):
	''' list participants with the lowest and highest tiv total, tiv gm, tiv wm and tiv lcr for
	manual checking of the volumes.

	Parameters
	----------
	df : dataframe including the TIV + the TIV of gm, wm and lcr per subject.
	
	percentile : int, optional
		percentile to which extract the most extreme TIV values
		Default=5.
	'''
	outliers = {}
	limit_above = 100 - percentile
	limit_below = percentile
	outliers['above_TIV'] = df['TIV'][df['TIV'] < np.percentile(df['TIV'], limit_below, interpolation = 'midpoint')]
	outliers['below_TIV'] = df['TIV'][df['TIV'] > np.percentile(df['TIV'], limit_above, interpolation = 'midpoint')]
	outliers['above_TIV_lcr'] = df['TIV_lcr'][df['TIV_lcr'] < np.percentile(df['TIV_lcr'], limit_below, interpolation = 'midpoint')]
	outliers['below_TIV_lcr'] = df['TIV_lcr'][df['TIV_lcr'] > np.percentile(df['TIV_lcr'], limit_above, interpolation = 'midpoint')]
	outliers['above_TIV_gm'] = df['TIV_gm'][df['TIV_gm'] < np.percentile(df['TIV_gm'], limit_below, interpolation = 'midpoint')]
	outliers['below_TIV_gm'] = df['TIV_gm'][df['TIV_gm'] > np.percentile(df['TIV_gm'], limit_above, interpolation = 'midpoint')]
	outliers['above_TIV_wm'] = df['TIV_wm'][df['TIV_wm'] < np.percentile(df['TIV_wm'], limit_below, interpolation = 'midpoint')]
	outliers['below_TIV_wm'] = df['TIV_wm'][df['TIV_wm'] > np.percentile(df['TIV_wm'], limit_above, interpolation = 'midpoint')]
	print(outliers)
	return outliers



################################ CLASSICAL STATISTICS TIV AGE AND SEX #########################


def plot_tif_info(df):
	''' Check and plot age/sex/tiv relationships.
	
	Parameters
	----------
	df : dataframe including TIV, TIV_gm, TIV_wm, TIV_lcr, Sex, and Age.

	'''
	age = df['Age']
	sex = df['Sex']
	sex[sex == 'M'] = 1
	sex[sex == 'F'] = 0
	tiv = df['TIV']
	tiv_gm = df['TIV_gm']
	tiv_wm = df['TIV_wm']
	tiv_lcr = df['TIV_lcr']


	# check age // tiv
	X = age.values.astype('float64')
	Y = tiv.values.astype('float64')
	# Create a data frame containing all the relevant variables
	data = pd.DataFrame({'x': X, 'y': Y})
	model = ols("y ~ x", data).fit()
	# print(model.summary())
	pvalAgeTIV = model.f_pvalue


	# check age // tiv_gm
	X = age.values.astype('float64')
	Y = tiv_gm.values.astype('float64')
	# Create a data frame containing all the relevant variables
	data = pd.DataFrame({'x': X, 'y': Y})
	model = ols("y ~ x", data).fit()
	# print(model.summary())
	pvalAgeTIVgm = model.f_pvalue

	# check age // tiv_wm
	X = age.values.astype('float64')
	Y = tiv_wm.values.astype('float64')
	# Create a data frame containing all the relevant variables
	data = pd.DataFrame({'x': X, 'y': Y})
	model = ols("y ~ x", data).fit()
	# print(model.summary())
	pvalAgeTIVwm = model.f_pvalue

	# check age // tiv_lcr
	X = age.values.astype('float64')
	Y = tiv_lcr.values.astype('float64')
	# Create a data frame containing all the relevant variables
	data = pd.DataFrame({'x': X, 'y': Y})
	model = ols("y ~ x", data).fit()
	# print(model.summary())
	pvalAgeTIVlcr = model.f_pvalue

	# check sex // tiv
	X = sex.values.astype(str)
	Y = tiv.values.astype('float64')
	# Create a data frame containing all the relevant variables
	data = pd.DataFrame({'x': X, 'y': Y})
	model = ols("y ~ C(x)", data).fit()
	# print(model.summary())
	pvalSexTIV = model.f_pvalue

	# check sex // tiv_gm
	X = sex.values.astype(str)
	Y = tiv_gm.values.astype('float64')
	# Create a data frame containing all the relevant variables
	data = pd.DataFrame({'x': X, 'y': Y})
	model = ols("y ~ C(x)", data).fit()
	# print(model.summary())
	pvalSexTIVgm = model.f_pvalue

	# check sex // tiv_wm
	X = sex.values.astype(str)
	Y = tiv_wm.values.astype('float64')
	# Create a data frame containing all the relevant variables
	data = pd.DataFrame({'x': X, 'y': Y})
	model = ols("y ~ C(x)", data).fit()
	# print(model.summary())
	pvalSexTIVwm = model.f_pvalue

	# check sex // tiv_lcr
	X = sex.values.astype(str)
	Y = tiv_lcr.values.astype('float64')
	# Create a data frame containing all the relevant variables
	data = pd.DataFrame({'x': X, 'y': Y})
	model = ols("y ~ C(x)", data).fit()
	# print(model.summary())
	pvalSexTIVlcr = model.f_pvalue


	# only age and TIV
	
	df['Sex'][df['Sex'] == 1] = 'M'
	df['Sex'][df['Sex'] == 0] = 'F'
	data = pd.DataFrame({'sex': df['Sex'].values.astype(str), 
						'Age':df['Age'].values.astype('float64'),
						'TIV': df['TIV'].values.astype('float64')})
	fig, ax = plt.subplots()
	ax = sns.pairplot(data, hue='sex', kind='reg')
	ax.fig.text(0.65, 0.85,"Age-TIV p={}".format(pvalAgeTIV), fontsize=9)
	ax.fig.text(0.65, 0.77,"Sex-TIV p={}".format(pvalSexTIV), fontsize=9)
	plt.savefig("eMCI/eMCI_agesextiv.png")
	plt.show()

	ax = sns.violinplot(x="sex", y="TIV",
	                    data=data, palette="Set2", split=True,
	                    scale="count", inner="stick", scale_hue=False)
	plt.savefig("eMCI/eMCI_sextiv.png")
	plt.show()

	# all in
	df['Sex'][df['Sex'] == 1] = 'M'
	df['Sex'][df['Sex'] == 0] = 'F'
	data = pd.DataFrame({'sex': df['Sex'].values.astype(str), 
						'Age':df['Age'].values.astype('float64'),
						'TIV_gm': df['TIV_gm'].values.astype('float64'),
						'TIV_wm': df['TIV_wm'].values.astype('float64'),
						'TIV_lcr': df['TIV_lcr'].values.astype('float64')})
	fig, ax = plt.subplots()
	ax = sns.pairplot(data, hue='sex', kind='reg')
	ax.fig.text(0.65, 0.83,"Age-TIV_gm p={}".format(pvalAgeTIVgm), fontsize=9)
	ax.fig.text(0.65, 0.81,"Age-TIV_wm p={}".format(pvalAgeTIVwm), fontsize=9)
	ax.fig.text(0.65, 0.79,"Age-TIV_lcr p={}".format(pvalAgeTIVlcr), fontsize=9)
	ax.fig.text(0.65, 0.75,"Sex-TIV_gm p={}".format(pvalSexTIVgm), fontsize=9)
	ax.fig.text(0.65, 0.73,"Sex-TIV_wm p={}".format(pvalSexTIVwm), fontsize=9)
	ax.fig.text(0.65, 0.71,"Sex-TIV_lcr p={}".format(pvalSexTIVlcr), fontsize=9)
	plt.savefig("eMCI/eMCI_agesextiv_details.png")
	plt.show()





def fit_atlas(atlas, df, saving_path=None, strategy='sum', show=0, labels=None):
	''' masking of the nifti of the participant cortex to extract ROI from atlas.
	
	Parameters
	----------
	atlas : mask used to extract ROIs
		must be file from nilearn.datasets.fetch_atlas_[...].

	df : dataframe including a 'greyMatter_path' column per subject (path to nifti cortex file).
	
	saving_path : str, optional
		if not None, save the dataframe with grey matter quantity per ROI per subject.

	strategy : define how the quantity of grey matter per voxel is summarized for each ROI
		Must be one of: sum, mean, median, minimum, maximum, variance, standard_deviation
		Default='sum'.
	show : 0 or 1
		plot the atlas onto the grey matter nifti file of the first subject to check the fit
		Default=0

	'''
	# extract grey matter volume for atlas ROI
	atlas_filename = atlas.maps
	# first subject nifti load to get affine and check mask fit
	tmp_nii = nib.load(df["greyMatter_path"].values[0])
	# show the mask fit onto a participant brain
	if show == 1:
		plotting.plot_img(tmp_nii).add_overlay(atlas_filename, cmap=plotting.cm.black_blue)
		plt.show()
	# shared affine for mask and cortex scan
	ratlas_nii = resample_img(
  		atlas.maps, target_affine=tmp_nii.affine, interpolation='nearest')
	# the grey matter volume will be stored in the FS list
	FS = []
	# iterate through each subject
	for i_nii, nii_path in enumerate(df["greyMatter_path"].values):
		# check memory
		print('The CPU usage is: ', psutil.cpu_percent(4))
		print(i_nii, ' / ', df["greyMatter_path"].__len__())
		# summarize grey matter quantity into each ROI
		nii = nib.load(nii_path)
		masker = NiftiLabelsMasker(labels_img=ratlas_nii, standardize=False, strategy=strategy)
		# extract each roi grey matter quantity as list (of length the number of ROIs)
		cur_FS = masker.fit_transform(nii)
		FS.append(cur_FS)
		print(cur_FS)
	FS = np.array(FS).squeeze()
	df_FS = pd.DataFrame(index=df.index, columns=labels, data=FS)
	# check quality of dataframization
	assert df_FS.iloc[-1].values.sum() == FS[-1].sum()
	# remove duplicates
	assert (df_FS.index.duplicated() == True).sum() == 0
	# save if asked
	if saving_path != None:
		df_FS.to_excel(saving_path)
	FS = None # empty memory
	return df_FS



def clean_signal(df, nb_rois, labels):
	''' clean signal using nilearn.signal.clean function.
	Regressed out variance that could be explained by the factors “age” and "TIV".
	
	Parameters
	----------
	df : pandas dataframe including the signal with ROIs numbered from 0 to len(nb_rois)
	plus a column for 'TIV' and for 'Age'.
	It should also include a column for 'TIV_gm', 'TIV_wm', 'TIV_lcr', 'Group', 'Sex'.

	nb_rois : the number of ROIs from the used atlas.
	
	'''
	# extract signal
	FS = df[labels].values
	# extract confound
	confounds = df[['TIV', 'Age']].values
	# clean signal from confound explained variance
	FS_cleaned = clean(FS, confounds=confounds, detrend=False)
	# restore into a dataframe
	df_cleaned = pd.DataFrame(columns=labels, index=df.index, data=FS_cleaned)
	info = df[['TIV', 'TIV_gm', 'TIV_wm', 'TIV_lcr', 'Group', 'Sex', 'Age', 'ADNI_MEM', 'ADNI_EF']]
	df_cleaned = merge_df_by_index(info, df_cleaned)
	# check quality of dataframization
	assert FS_cleaned[0][10] == df_cleaned[labels[10]].iloc[0]
	return df_cleaned





def plot_matrice(df, labels, title, saving_path=None, show=1):
	''' Plot the matrices
	
	Parameters
	----------
	df : pandas dataframe including the signal with ROIs numbered from 0 to len(nb_rois)

	labels : lst
		atlas ROIs name

	title : str
		Title for the plot
	
	saving_path : str
		Path to save the covariance matrix plot
	
	show : 0 or 1
		plot the matrices
		Default=1

	'''
	sns.set_theme(style="white")
	mask = np.triu(np.ones_like(df, dtype=bool))
	f, ax = plt.subplots(figsize=(11, 9))
	# Generate a custom diverging colormap
	cmap = sns.diverging_palette(230, 20, as_cmap=True)
	# Draw the heatmap with the mask and correct aspect ratio
	sns.heatmap(df, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, xticklabels=True, yticklabels=True, cbar_kws={"shrink": .5})
	plt.title(title)
	plt.tight_layout()
	if saving_path != None:
		plt.savefig(saving_path)
	if show == 1:
		plt.show()


def louvainize(df_cov, title,saving_path=None):
	''' Compute network graph, then louvain community
	and reorganized it into a matrix, and plot it

	Parameters
	----------
	df_cov : pandas dataframe of the covariance matrix (n_roi*n_roi)

	saving_path : str
		if not None, saved at the given path

	title : str
		title for thz final matrix

	'''
	# compute the best partition
	G = nx.from_numpy_matrix(df_cov.values)  
	nx.draw(G, with_labels=True) 
	partition = community.best_partition(G, random_state=0)
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
	# check positionning from original matrix to louvain matri
	# get index of first roi linked to community 0
	index_roi_com0_louvain = list(partition.values()).index(0)
	# get nb of roi in community 0
	nb_com0 = np.unique(list(partition.values()), return_counts=True)[1][0]
	# # get index of first roi linked to community 1
	index_roi_com1_louvain = list(partition.values()).index(1)
	assert louvain[0].sum() == df_cov.values[index_roi_com0_louvain].sum()
	assert louvain[nb_com0].sum() == df_cov.values[index_roi_com1_louvain].sum() 

	df_louvain = pd.DataFrame(index=labels_new_order, columns=labels_new_order, data=louvain)
	df_louvain.to_excel("eMCI/df_{}.xlsx".format(title))
	plot_matrice(df_louvain, labels_new_order, title, saving_path=saving_path, show=0)
	

def return_all_plot(df_FS):
	''' Compute covariance matrices and plot them
	
	Parameters
	----------
	df_FS : dataframe of the time series per ROI per subject.

	'''
	
	# # clean for age and tiv
	# df_FS_cleaned = clean_signal(df_FS, nb_rois, labels)

	# ledoiwolf covariance
	labels = df_FS.columns
	matrix = LedoitWolf().fit(df_FS)
	cov = matrix.covariance_
	df_ledoit_cov = pd.DataFrame(index=labels, columns=labels, data=cov)
	df_ledoit_cov.to_excel("eMCI/ledoiwolf_cov.xlsx")
	plot_matrice(df_ledoit_cov, labels, "ledoiwolf_cov", saving_path="eMCI/ledoiwolf_cov.png", show=0)
	louvainize(df_ledoit_cov, "Louvain_LedoitWolf", "eMCI/Louvain_LedoitWolf.png")



	prec = matrix.precision_
	df_prec = pd.DataFrame(index=labels, columns=labels, data=prec)
	df_prec.to_excel("eMCI/ledoiwolf_prec.xlsx")
	plot_matrice(df_prec, labels, "ledoiwolf_prec", saving_path="eMCI/ledoiwolf_prec.png", show=0)
	louvainize(df_prec, "Louvain_LedoitWolf_prec", "eMCI/Louvain_LedoitWolf_prec.png")

	# pearson
	pearson = np.corrcoef(df_FS.values.T)
	df_pearson = pd.DataFrame(index=labels, columns=labels, data=pearson)
	df_pearson.to_excel("eMCI/pearson.xlsx")
	plot_matrice(df_pearson, labels, "pearson", saving_path="eMCI/pearson.png", show=0)
	louvainize(df_pearson, "Louvain_Pearson", "eMCI/Louvain_Pearson.png")

	# covariance
	cov = np.cov(df_FS.values.T)
	df_cov = pd.DataFrame(index=labels, columns=labels, data=cov)
	df_cov.to_excel("eMCI/cov.xlsx")
	plot_matrice(df_cov, labels, "cov", saving_path="eMCI/cov.png", show=0)
	louvainize(df_cov, "Louvain_cov", "eMCI/Louvain_cov.png")


# extract all file from folder
files = glob.glob('C:\\Users\\lefortb211\\Downloads\\ADNI_eMCI_smwc\\*')
# Sort and store nifti paths of gm, wm, and lcr into a dataframe
df = dataframize_nii_paths(files)
# compute TIV and store it in the df
df = compute_store_tiv(df)
# list most extrame values for TIV, tiv gm, tiv wm, tiv lcr
outliers = list_outliers(df)
# add age group and sex information
df_demog = pd.read_csv("eMCI/eMCI_3T_6_23_2021.csv")
df_demog = df_demog[['Subject', 'Group', 'Sex', 'Age']] # subject, group, sex, age
df_demog = df_demog.set_index('Subject')
df = merge_df_by_index(df, df_demog)
# add executive scores (file -containing each diag info- in CN folder to avoid duplicates)
df_executive = pd.read_excel("CN/score_ex.xlsx") 
df_executive = df_executive.set_index('PTID')
df = merge_df_by_index(df, df_executive)
# plot relationships between tiv age and sex
plot_tif_info(df)

atlas = ds.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm',  symmetric_split=True)
labels = atlas.labels[1:]
nb_rois = len(labels)
df_FS = fit_atlas(atlas, df, saving_path='eMCI/df_FS.xlsx', labels=labels)
# standardize values
df_FS_ss = pd.DataFrame(columns=df_FS.columns, index=df_FS.index, data=StandardScaler().fit_transform(df_FS.values))
info = df[['TIV', 'TIV_gm', 'TIV_wm', 'TIV_lcr', 'Group', 'Sex', 'Age', 'ADNI_MEM', 'ADNI_EF']]
df = merge_df_by_index(info, df_FS_ss)
return_all_plot(df_FS_ss)
plt.close('all')




###############################
### CHECKING LOUVAIN COMMUs ###
###############################

def niftiise_louvain_community(df_cov, saving_path=None):
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
	atlas_nii = atlas.maps
	voxelData = atlas_nii.get_data()
	for ind, label in enumerate(labels):
		voxelData[voxelData == ind + 1] = partition[ind] + 100 # +100 to prevent from mix between roi value and partition number
	nb_partition = np.unique(list(partition.values())).__len__()
	for nb in range(nb_partition):
		voxelData[voxelData == 100 + nb] = nb + 1 # get back to partition number from 1 to nb of partition
	# save a nifiti image
	partition_nifti = nib.Nifti1Image(
		voxelData, affine=atlas_nii.affine)
	partition_nifti.to_filename(saving_path) # transform as nii and save

	print("nb of partition = ", nb_partition)
	print("unique partition in final nifti file = ", np.unique(voxelData))

df_ledoiwolf_cov = pd.read_excel("eMCI/ledoiwolf_cov.xlsx")
df_ledoiwolf_cov = df_ledoiwolf_cov.set_index('Unnamed: 0')
niftiise_louvain_community(df_ledoiwolf_cov, saving_path="eMCI/ledoiwolf_cov_partition_nifti.nii")

df_ledoiwolf_prec = pd.read_excel("eMCI/ledoiwolf_prec.xlsx")
df_ledoiwolf_prec = df_ledoiwolf_prec.set_index('Unnamed: 0')
niftiise_louvain_community(df_ledoiwolf_prec, saving_path="eMCI/ledoiwolf_prec_partition_nifti.nii")

df_pearson = pd.read_excel("eMCI/pearson.xlsx")
df_pearson = df_pearson.set_index('Unnamed: 0')
niftiise_louvain_community(df_pearson, saving_path="eMCI/pearson_partition_nifti.nii")

df_cov = pd.read_excel("eMCI/cov.xlsx")
df_cov = df_cov.set_index('Unnamed: 0')
niftiise_louvain_community(df_cov, saving_path="eMCI/cov_partition_nifti.nii")
