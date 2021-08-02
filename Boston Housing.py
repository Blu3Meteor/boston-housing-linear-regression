import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_boston

# loading the dataset
boston_dataset = load_boston()

# building a dataframe
boston = pd.DataFrame(boston_dataset.data, columns = boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target

# print(boston.shape)

# # dataset description
# print(boston_dataset.DESCR)

# # summary stats
# print(boston.describe().round(2))

## some relevant visualizations ##

# boston.hist(column = 'CRIM')
# plt.savefig("CRIM hist.png")
# plt.show()
## CRIM appears to be right (positively) skewed, with most values between 0 and 9%
## Summary stats show that the median is much lesser than the mean, affirming the above

# boston.hist(column = 'ZN')
# plt.savefig("ZN hist.png")
# plt.show()
## Summary stats show that ZN's mean is much higher than its median
## Right (positive) skewness confirmed by graph, so proportion of land zoned is less

# boston.hist(column = 'INDUS')
# plt.savefig("INDUS hist.png")
# plt.show()
## INDUS' mean is slightly greater than its median
## Slight right (positive) skew confirmed by graph, following a bimodal distribution

# boston.hist(column = 'CHAS')
# plt.savefig("CHAS hist.png")
# plt.show()
## CHAS is mostly 0's (over 450 out of 506 values) so very few houses near the river

# boston.hist(column = 'NOX')
# plt.savefig("NOX hist.png")
# plt.show()
## NOX has similar mean and median
## Histogram shows a rough normal dist

# boston.hist(column = 'RM')
# plt.savefig("RM hist.png")
# plt.show()
## RM appears to be normally distributed, the summary stats show median and mean are similar
## the histogram plot affirms the same, so the average number of rooms is normally distributed

## end of visualization code ##



