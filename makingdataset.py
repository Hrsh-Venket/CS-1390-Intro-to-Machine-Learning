import numpy as np
import pandas as pd

# generate a dataset of size (50000, 5000)
X = np.random.rand(1000, 100)

# convert the dataset to a pandas DataFrame
df = pd.DataFrame(X)

# save the DataFrame as a CSV file
df.to_csv('massivedataset.csv', index=False)
