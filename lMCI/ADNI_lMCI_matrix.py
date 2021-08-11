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

np.random.seed(0)

# ############### moving zipped file to a folder ##########
# import glob
# import os

# files = glob.glob('C:\\Users\\lefortb211\\Downloads\\ADNI_lMCI\\*\\*\\*\\*\\smw*')
# for ind, file in enumerate(files):
# 	print(ind, '/', files.__len__())
# 	img_name = file.split('\\')[-1]
# 	new_path = "C:\\Users\\lefortb211\\Downloads\\ADNI_lMCI_dartels\\" + img_name
# 	try:
# 		os.rename(file, new_path)
# 	except:
# 		print("problem with ", file)

print("******")
print("EXTRACTING TIV FROM NIFTI FILES")
print("******")
time.sleep(3)
############### extract TIV from images ##########
files = glob.glob('C:/users/lefortb211/downloads/ADNI_lMCI_dartels/*')

# extract path nii files of greyMatter, whiteMatter and LCR per subject
# store it in dictionnary
files_per_sub = {}
for file in files:
	split_name = file.split('_')
	sub_nb = split_name[3] + '_' + split_name[4] + '_' + split_name[5]
	# print(sub_nb)
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
	# print(sub_nb, " Done")
	files_per_sub[sub_nb] = [smwc1, smwc2, smwc3]

# create a df from paths
df_lmci = pd.DataFrame.from_dict(files_per_sub, orient='index')

# rename columns accordingly
df_lmci.rename(columns={
            0: 'greyMatter',
            1: 'whiteMatter',
            2: 'LCR'}, inplace=True)



# compute TIV and store it in the df
df_lmci['TIV'] = '*'
df_lmci['TIV_gm'] = '*'
df_lmci['TIV_wm'] = '*'
df_lmci['TIV_lcr'] = '*'
dim_expected = nib.load(df_lmci['greyMatter'].iloc[0]).header.values()[15]
for row in df_lmci.index:
	smwc1 = nib.load(df_lmci['greyMatter'].loc[row])
	smwc2 = nib.load(df_lmci['whiteMatter'].loc[row])
	smwc3 = nib.load(df_lmci['LCR'].loc[row])
	# check voxel dimension 
	assert smwc1.header.values()[15].sum() == dim_expected.sum()
	assert smwc2.header.values()[15].sum() == dim_expected.sum()
	assert smwc3.header.values()[15].sum() == dim_expected.sum()
	# compute TIV 
	tiv1 = smwc1.get_data().sum()/1000
	tiv2 = smwc2.get_data().sum()/1000
	tiv3 = smwc3.get_data().sum()/1000
	TIV = tiv1 + tiv2 + tiv3
	print("Sub ", row, " TIV = ", TIV)
	# store TIV in df
	df_lmci['TIV_gm'].loc[row] = tiv1
	df_lmci['TIV_wm'].loc[row] = tiv2
	df_lmci['TIV_lcr'].loc[row] = tiv3
	df_lmci['TIV'].loc[row] = TIV

# test
assert df_lmci.iloc[-1]['TIV_wm'] == tiv2
assert (df_lmci.TIV == 0).sum() == 0

# sub_to_drop = ['137_S_4227']
# df_lmci = df_lmci.drop(sub_to_drop)
# df_lmci = df_lmci[df_lmci.index.duplicated() == False]

print("******")
print("TIV SAVED")
print("******")
print(df_lmci.head())
time.sleep(1)
print("******")
print("Extracting grey matter volum per roi")
print("******")
time.sleep(3)

# extract roi per subject lmci
dataset = ds.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
atlas_filename = dataset.maps
labels = dataset.labels
# extracting data MRI

# make sure the maps are onto the brain, otherwise need resampling
nii = df_lmci["greyMatter"].values[0]
from nilearn import plotting, image
plotting.plot_img(nib.load(nii)).add_overlay(atlas_filename, cmap=plotting.cm.black_blue)
plt.show()

# extract grey matter volume for atlas ROI
FS = []
for i_nii, nii_path in enumerate(df_lmci["greyMatter"].values):
	print('The CPU usage is: ', psutil.cpu_percent(4))
	print(i_nii, ' / ', df_lmci["greyMatter"].__len__())
	nii = nib.load(nii_path)
	nii_fake4D = nib.Nifti1Image(
		nii.get_data()[:, :, :, None], affine=nii.affine)
	masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True,
                           memory='nilearn_cache', verbose=5, )
	cur_FS = masker.fit_transform(nii_fake4D)
	FS.append(cur_FS)

assert len(df_lmci["greyMatter"].values) == len(FS)
FS = np.array(FS).squeeze()

# FS = np.load("work/ADNI_dartel/FS_lmci_DARTEL.npy")
df_FS = pd.DataFrame(index=df_lmci.index, data=FS)
# test
assert df_FS.iloc[-1].values.sum() == FS[-1].sum()
assert (df_FS.index.duplicated() == True).sum() == 0

# save info
df_FS.to_excel("ADNI_covariance_project/lMCI/df_fs.xlsx")
FS=None # empty memory

print("******")
print("volume extracted")
print("******")
print(df_FS.head())
time.sleep(1)
print("******")
print("Checking age and sex relationship with TIV")
print("******")
time.sleep(3)



# link df with demographic info
df_demog = pd.read_csv("ADNI_covariance_project/lMCI/lMCI_3T_6_23_2021.csv")
df_demog = df_demog[df_demog.columns[[1, 2, 3, 4]]]
df_merged = df_lmci.merge(df_demog, left_on=df_lmci.index, right_on=df_demog['Subject'])
df_merged = df_merged.set_index('key_0')
df_merged = df_merged[df_merged.index.duplicated() == False]

# check merging
assert (df_merged.index.duplicated() == True).sum() == 0
assert (df_merged['Subject'] == df_merged.index).sum() == df_merged.__len__()
assert (df_merged.TIV == 0).sum() == 0

################################ CLASSICAL STATISTICS TIV AGE AND SEX #########################

age = df_merged['Age']
sex = df_merged['Sex']
sex[sex == 'M'] = 1
sex[sex == 'F'] = 0
tiv = df_merged['TIV']
tiv_gm = df_merged['TIV_gm']
tiv_wm = df_merged['TIV_wm']
tiv_lcr = df_merged['TIV_lcr']

from statsmodels.formula.api import ols
# check age // tiv
X = age.values.astype('float64')
Y = tiv.values.astype('float64')
# Create a data frame containing all the relevant variables
data = pd.DataFrame({'x': X, 'y': Y})
model = ols("y ~ x", data).fit()
print(model.summary())
pvalAgeTIV = model.f_pvalue


# check age // tiv_gm
X = age.values.astype('float64')
Y = tiv_gm.values.astype('float64')
# Create a data frame containing all the relevant variables
data = pd.DataFrame({'x': X, 'y': Y})
model = ols("y ~ x", data).fit()
print(model.summary())
pvalAgeTIVgm = model.f_pvalue

# check age // tiv_wm
X = age.values.astype('float64')
Y = tiv_wm.values.astype('float64')
# Create a data frame containing all the relevant variables
data = pd.DataFrame({'x': X, 'y': Y})
model = ols("y ~ x", data).fit()
print(model.summary())
pvalAgeTIVwm = model.f_pvalue

# check age // tiv_lcr
X = age.values.astype('float64')
Y = tiv_lcr.values.astype('float64')
# Create a data frame containing all the relevant variables
data = pd.DataFrame({'x': X, 'y': Y})
model = ols("y ~ x", data).fit()
print(model.summary())
pvalAgeTIVlcr = model.f_pvalue

# check sex // tiv
X = sex.values.astype(str)
Y = tiv.values.astype('float64')
# Create a data frame containing all the relevant variables
data = pd.DataFrame({'x': X, 'y': Y})
model = ols("y ~ C(x)", data).fit()
print(model.summary())
pvalSexTIV = model.f_pvalue

# check sex // tiv_gm
X = sex.values.astype(str)
Y = tiv_gm.values.astype('float64')
# Create a data frame containing all the relevant variables
data = pd.DataFrame({'x': X, 'y': Y})
model = ols("y ~ C(x)", data).fit()
print(model.summary())
pvalSexTIVgm = model.f_pvalue

# check sex // tiv_wm
X = sex.values.astype(str)
Y = tiv_wm.values.astype('float64')
# Create a data frame containing all the relevant variables
data = pd.DataFrame({'x': X, 'y': Y})
model = ols("y ~ C(x)", data).fit()
print(model.summary())
pvalSexTIVwm = model.f_pvalue

# check sex // tiv_lcr
X = sex.values.astype(str)
Y = tiv_lcr.values.astype('float64')
# Create a data frame containing all the relevant variables
data = pd.DataFrame({'x': X, 'y': Y})
model = ols("y ~ C(x)", data).fit()
print(model.summary())
pvalSexTIVlcr = model.f_pvalue


# only age and TIV
import seaborn as sns
df_merged['Sex'][df_merged['Sex'] == 1] = 'M'
df_merged['Sex'][df_merged['Sex'] == 0] = 'F'
data = pd.DataFrame({'sex': df_merged['Sex'].values.astype(str), 'Age':df_merged['Age'].values.astype('float64'),
					'TIV': df_merged['TIV'].values.astype('float64')})
fig, ax = plt.subplots()
ax = sns.pairplot(data, hue='sex', kind='reg')
ax.fig.text(0.85, 0.85,"Age-TIV p={}".format(pvalAgeTIV), fontsize=9)
ax.fig.text(0.85, 0.77,"Sex-TIV p={}".format(pvalSexTIV), fontsize=9)
plt.savefig("ADNI_covariance_project/lMCI/lmci_agesextiv.png")
plt.show()

ax = sns.violinplot(x="sex", y="TIV",
                    data=data, palette="Set2", split=True,
                    scale="count", inner="stick", scale_hue=False)
plt.savefig("ADNI_covariance_project/lMCI/lmci_sextiv.png")
plt.show()

# all in
import seaborn as sns
df_merged['Sex'][df_merged['Sex'] == 1] = 'M'
df_merged['Sex'][df_merged['Sex'] == 0] = 'F'
data = pd.DataFrame({'sex': df_merged['Sex'].values.astype(str), 
					'Age':df_merged['Age'].values.astype('float64'),
					'TIV_gm': df_merged['TIV_gm'].values.astype('float64'),
					'TIV_wm': df_merged['TIV_wm'].values.astype('float64'),
					'TIV_lcr': df_merged['TIV_lcr'].values.astype('float64')})
fig, ax = plt.subplots()
ax = sns.pairplot(data, hue='sex', kind='reg')
ax.fig.text(0.85, 0.83,"Age-TIV_gm p={}".format(pvalAgeTIVgm), fontsize=9)
ax.fig.text(0.85, 0.81,"Age-TIV_wm p={}".format(pvalAgeTIVwm), fontsize=9)
ax.fig.text(0.85, 0.79,"Age-TIV_lcr p={}".format(pvalAgeTIVlcr), fontsize=9)
ax.fig.text(0.85, 0.75,"Sex-TIV_gm p={}".format(pvalSexTIVgm), fontsize=9)
ax.fig.text(0.85, 0.73,"Sex-TIV_wm p={}".format(pvalSexTIVwm), fontsize=9)
ax.fig.text(0.85, 0.71,"Sex-TIV_lcr p={}".format(pvalSexTIVlcr), fontsize=9)
plt.savefig("ADNI_covariance_project/lMCI/lmci_agesextiv_details.png")
plt.show()


# lcr>800
# wm>550 et <280
# gm>710 et <460
# tiv<1100


########################## COVARIANCE MATRIX ################

# covariance analysis
# load grey matter quatities associated with subject name
df_FS = pd.read_excel("ADNI_covariance_project/lMCI/df_fs.xlsx")
df_FS = df_FS.set_index('Unnamed: 0')

# clean signal for age and tiv
df_FS_cleaned = df_FS.join(df_merged[['TIV', 'Age']], how='outer')
FS = df_FS_cleaned[df_FS_cleaned.columns[:-2]].values
confounds = df_FS_cleaned[df_FS_cleaned.columns[-2:]].values
FS_cleaned = clean(FS, confounds=confounds, detrend=False) 
df_FS_cleaned[df_FS_cleaned.columns[:-2]] = FS_cleaned
df_FS_cleaned = df_FS_cleaned[df_FS_cleaned.columns[:-2]]
assert FS_cleaned[0][10] == df_FS_cleaned[10].iloc[0]
assert FS_cleaned[10][0] == df_FS_cleaned[0].iloc[10]

# link df with mri info with demographic info
df_allinfo = pd.read_excel("ADNI_covariance_project/CN/score_ex.xlsx") # this df is in CN folder
df_allinfo = df_allinfo.set_index('PTID')
df_allinfo = df_allinfo.merge(df_merged, left_on=df_allinfo.index, right_on=df_merged.index)
df_allinfo = df_allinfo.set_index('key_0')

# check merging
# assert df_allinfo.index.__len__() == df_FS.__len__()
assert (df_allinfo.index == df_allinfo['Subject'].values).sum() == df_allinfo.__len__() 

# remove double participants
nb_dup = df_allinfo.__len__() - (df_allinfo.index.duplicated() == False).sum()
df_allinfo = df_allinfo[df_allinfo.index.duplicated() == False]


print("{} participants were duplicates".format(nb_dup))
time.sleep(2)


# separate into low and high cognitive profiles
df_lmci_high = df_allinfo[df_allinfo['ADNI_EF']>=df_allinfo['ADNI_EF'].median()]
FS_high = df_FS.loc[df_lmci_high.index].values
# Age_high = df_lmci_high['Age'].values
# TIV_high = df_lmci_high['TIV'].values

df_lmci_low = df_allinfo[df_allinfo['ADNI_EF']<df_allinfo['ADNI_EF'].median()]
FS_low = df_FS.loc[df_lmci_low.index].values
# Age_low = df_lmci_low['Age'].values
# TIV_low = df_lmci_low['TIV'].values



from sklearn.covariance import LedoitWolf
import seaborn as sns
# compute precision matrix for high cognitive level
matrix = LedoitWolf(store_precision=True).fit(FS_high)
prec_high = matrix.precision_
cov_high = matrix.covariance_
corr = pd.DataFrame(index=labels[1:], columns=labels[1:], data=prec_high)
#plot matrices
sns.set_theme(style="white")
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, xticklabels=True, yticklabels=True, cbar_kws={"shrink": .5})
plt.savefig("ADNI_covariance_project/lMCI/lmci_prec_high.png")
plt.show()

# compute precision matrix for low cognitive level
matrix = LedoitWolf(store_precision=True).fit(FS_low)
prec = matrix.precision_
cov = matrix.covariance_
corr = pd.DataFrame(index=labels[1:], columns=labels[1:], data=prec)
#plot matrices
sns.set_theme(style="white")
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, xticklabels=True, yticklabels=True, cbar_kws={"shrink": .5})
plt.savefig("ADNI_covariance_project/lMCI/lmci_prec_low.png")
plt.show()

########################## non parametric hypothesis testing ###############

np.random.seed(0)
# non-parametric multivariate hypothesis testing by bootstrapping
h_precs, h_covs = [], []
n_perm = 1000  # p < 0.05
n_low_samples = len(FS_low)
for p in range(n_perm):
    print('Bootstrapping iteration: {}/{}'.format(p + 1, n_perm))
    new_inds = np.random.randint(0, n_low_samples, n_low_samples)
    bs_sample = FS_low[new_inds]

    gsc_low = LedoitWolf(store_precision=True).fit(bs_sample)
    h_covs.append(gsc_low.covariance_)
    h_precs.append(gsc_low.precision_)

tril_inds = np.tril_indices_from(h_covs[0])
covs_ravel = np.array([cov[tril_inds] for cov in h_covs])
precs_ravel = np.array([prec[tril_inds] for prec in h_precs])

cov_test_high = cov_high[tril_inds]
prec_test_high = prec_high[tril_inds]

margin = (1. / n_perm) * 100 / 2

cov_sign1 = cov_test_high < np.percentile(covs_ravel, margin, axis=0)
cov_sign2 = cov_test_high > np.percentile(covs_ravel, 100 - margin, axis=0)
cov_sign = np.zeros_like(cov_high)
cov_sign[tril_inds] = np.logical_or(cov_sign1, cov_sign2)

prec_sign1 = prec_test_high < np.percentile(precs_ravel, margin, axis=0)
prec_sign2 = prec_test_high > np.percentile(precs_ravel, 100 - margin, axis=0)
prec_sign = np.zeros_like(prec_high)
prec_sign[tril_inds] = np.logical_or(prec_sign1, prec_sign2)


corr = pd.DataFrame(index=labels[1:], columns=labels[1:], data=prec_sign)
#plot matrices
sns.set_theme(style="white")
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(10, 10))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, xticklabels=True, yticklabels=True, cbar=False, ax=ax)

plt.title("Precision matrice differences high and low lMCI (p<0.05)")
plt.tight_layout()
plt.savefig("ADNI_covariance_project/lMCI/lmci_prec_diff.png")
plt.show()

