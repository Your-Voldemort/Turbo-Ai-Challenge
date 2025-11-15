import pandas as pd

df = pd.read_csv('Comp_Hotel_competition.csv')
print('Columns:', df.columns.tolist())
print('\nShape:', df.shape)
print('\nFirst few rows:')
print(df.head())
