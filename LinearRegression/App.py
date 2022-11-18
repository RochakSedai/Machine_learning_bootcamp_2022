import pandas as pd 
import numpy as np

# read .csv into DataFrame
house_data = pd.read_csv('house_prices.csv')
size = house_data['sqft_living']
price = house_data['price']
print(price)

#machine learning handle array not data frame
x = np.array(size).reshape(-1,1)
y = np.array(price).reshape(-1,1)

# ma