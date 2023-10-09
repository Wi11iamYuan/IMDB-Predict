# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
import ast
from functools import lru_cache

# %%
from xgboost import XGBRegressor
# %%
original_df = pd.read_csv('movies.csv')
df = original_df.copy()
df

# %%
# count missing values
df.isna().sum()

# %%
# convert director and stars columns to lists
df['director'] = df['director'].apply(ast.literal_eval)
df['stars'] = df['stars'].apply(ast.literal_eval)
df['genre'] = df['genre'].str.split(', ')

# %%
# make sure all movies have a rating
df['certificate'].fillna('Unrated', inplace=True)

# %%
# drop title
df.drop(['title'], axis=1, inplace=True)

# %%
# replace year with last 4 digits
df['year'] = df['year'].str[-4:].astype(int)

# %%
# interpolate gross with linear regression on metascore
m = df['gross'].corr(df['metascore']) * df['gross'].std() / df['metascore'].std()
b = df['gross'].mean() - m * df['metascore'].mean()
df['gross'] = df['gross'].fillna(df['metascore'] * m + b)

# %%
# interpolate metascore with median
df['metascore'] = df['metascore'].fillna(df['metascore'].median())


# %%
# one hot encode certificate
ohe = OneHotEncoder()
certificate = ohe.fit_transform(df[['certificate']]).astype(int).toarray()
certificate_df = pd.DataFrame(certificate, columns=['certificate_' + col for col in ohe.categories_[0]])
df = pd.concat([df, certificate_df], axis=1)

df.drop(['certificate'], axis=1, inplace=True)
df

# %%
def ohe_list(df, col: str):
	temp_df = df[[col]].copy()

	temp_df = temp_df.explode(col)
	temp_df = pd.get_dummies(temp_df, columns=[col], prefix=col, prefix_sep='_')

	temp_df = temp_df.groupby(temp_df.index).sum()

	return temp_df


# %%
genre_df = ohe_list(df, 'genre')
df = df.join(genre_df)
df.drop('genre', axis=1, inplace=True)
del genre_df

# %%
# dataframe with all directors and the number of movies they have directed
director_counts = {}
for row in df['director']:
	for director in row:
		director_counts[director] = director_counts.get(director, 0) + 1

director_scores = pd.DataFrame(list(director_counts.items()), columns=['Director', 'Score']).sort_values(by='Score', ascending=False)
director_scores

# %%
# new column 'Director Score' that is the sum of the scores of each director
@lru_cache(maxsize=None)
def get_director_score(director):
	return director_scores.loc[director_scores['Director'] == director, 'Score'].values[0]

df['Director Score'] = df['director'].apply(lambda x: sum([get_director_score(director) for director in x]))
df.drop('director', axis=1, inplace=True)

# %%
# dataframe with all stars and the number of movies they have starred in
star_counts = {}
for row in df['stars']:
	for star in row:
		star_counts[star] = star_counts.get(star, 0) + 1

star_scores = pd.DataFrame(list(star_counts.items()), columns=['Star', 'Score']).sort_values(by='Score', ascending=False)
star_scores

# %%
# new column 'Star Score' that is the sum of the scores of each star
@lru_cache(maxsize=None)
def get_star_score(star):
	return star_scores.loc[star_scores['Star'] == star, 'Score'].values[0]

df['Star Score'] = df['stars'].apply(lambda x: sum([get_star_score(star) for star in x]))
df.drop('stars', axis=1, inplace=True)

# %%
# standardize all columns (u=0, std=1)
scaler = StandardScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])

# %%
# get indpedendent and dependent variables
x = df.drop(['rating'], axis=1)
y = df['rating']

# %%
# split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y)

# %%
model = XGBRegressor()
model.fit(x_train, y_train)

# %%
y_pred = model.predict(x_test)
r2_score(y_test, y_pred)

# %%
# plot 10 most important features
feature_importances = pd.DataFrame(model.feature_importances_, index=x_train.columns, columns=['importance']).sort_values(by='importance', ascending=False)
feature_importances[:10].plot.bar()
plt.show()

