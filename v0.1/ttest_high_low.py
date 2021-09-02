from scipy.stats import ttest_ind

def sign(ttest):
	if ttest[1] < (0.05**12): # Bonferroni corrected
		print("SIGNIFICATIF")
	else:
		print("NON SIGNIFICATIF")


sign(ttest_ind(dic['SMC'][0], dic['CN'][0]))
sign(ttest_ind(dic['SMC'][0], dic['eMCI'][0]))
sign(ttest_ind(dic['SMC'][0], dic['lMCI'][0]))
sign(ttest_ind(dic['CN'][0], dic['eMCI'][0]))
sign(ttest_ind(dic['CN'][0], dic['lMCI'][0]))
sign(ttest_ind(dic['eMCI'][0], dic['lMCI'][0]))


sign(ttest_ind(dic['SMC'][1], dic['CN'][1]))
sign(ttest_ind(dic['SMC'][1], dic['eMCI'][1]))
sign(ttest_ind(dic['SMC'][1], dic['lMCI'][1]))
sign(ttest_ind(dic['CN'][1], dic['eMCI'][1]))
sign(ttest_ind(dic['CN'][1], dic['lMCI'][1]))
sign(ttest_ind(dic['eMCI'][1], dic['lMCI'][1]))

# plot it high and then low
all_data = [dic['CN'][0], dic['SMC'][0], dic['eMCI'][0], dic['lMCI'][0],
			dic['CN'][1], dic['SMC'][1], dic['eMCI'][1], dic['lMCI'][1]]
labels = ['CN_high', 'SMC_high', 'eMCI_high', 'lMCI_high',
		'CN_low', 'SMC_low', 'eMCI_low', 'lMCI_low',
		]

# plot it high low and then next diag
all_data = [dic['CN'][0], dic['CN'][1], dic['SMC'][0], dic['SMC'][1],
		dic['eMCI'][0], dic['eMCI'][1], dic['lMCI'][0], dic['lMCI'][1]]
labels = ['CN_high', 'CN_low', 'SMC_high', 'SMC_low',
		'eMCI_high', 'eMCI_low', 'lMCI_high', 'lMCI_low',
		]


fig, (ax) = plt.subplots(figsize=(9, 4))


# rectangular box plot
bplot = ax.boxplot(all_data,
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labels)  # will be used to label x-ticks

colors = ['lightpink', 'palegoldenrod'] * 4 
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
plt.axvline(x=4.5)
plt.title('Bonferroni corrected : No significative results')

plt.show()