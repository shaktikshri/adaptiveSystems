import numpy as np
from scipy import stats

# In[]:
# Defining 2 random distributions with sample size 10
# Gaussian distributed data with mean = 2 and var = 1
Nx = 10
x = np.random.randn(Nx) + 2
# Gaussian distributed data with with mean = 0 and var = 1
Ny = 10
y = np.random.randn(Ny)

# In[]:

# Calculate the std dev now
# ddof : int, optional
# “Delta Degrees of Freedom”: the divisor used in the calculation is N - ddof,
# where N represents the number of elements. By default ddof is zero.
var_x = x.var(ddof=1)
var_y = y.var(ddof=1)

# In[]:
# Calculate the t-statistics
t = (x.mean() - y.mean()) / (np.sqrt( (var_x/Nx) + (var_y/Ny) ))

# In[]:
# Compare with the critical t-value

# Degrees of freedom
df = Nx + Ny - 2
# p-value after comparison with the t
p = 1 - stats.t.cdf(t,df=df)

# In[]:
print("t = " + str(t))
print("p = " + str(2*p))
### You can see that after comparing the t statistic with the critical t value (computed internally) we get a good p value of 0.0005 and thus we reject the null hypothesis and thus it proves that the mean of the two distributions are different and statistically significant.


## Cross Checking with the internal scipy function
t2, p2 = stats.ttest_ind(a,b)
print("t = " + str(t2))
print("p = " + str(p2))
