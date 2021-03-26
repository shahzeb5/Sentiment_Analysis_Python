import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

youtube_data = pd.read_csv('video_result.csv')


print(youtube_data.corr())

plt.scatter(youtube_data.viewCount,youtube_data.likeCount)

y = youtube_data.likeCount
X = youtube_data.viewCount
X = sm.add_constant(X)

lr_model = sm.OLS(y,X).fit()

print(lr_model.summary())

X_prime =  np.linspace(X.viewCount.min(),X.viewCount.max(),100)
X_prime = sm.add_constant(X_prime)

y_hat = lr_model.predict(X_prime)
plt.scatter(X.viewCount,y)
plt.xlabel("View Count")
plt.ylabel("Like Count")
plt.plot(X_prime[:,1],y_hat)