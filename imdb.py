# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
df['certificate'].fillna('Not Rated', inplace=True)

# %%
# drop title
df.drop(['title','votes'], axis=1, inplace=True)

# %%
# replace year with last 4 digits
df['year'] = df['year'].str[-4:].astype(int)

# %%
# interpolate gross with linear regression on metascore
# m = df['gross'].corr(df['metascore']) * df['gross'].std() / df['metascore'].std()
# b = df['gross'].mean() - m * df['metascore'].mean()
# df['gross'] = df['gross'].fillna(df['metascore'] * m + b)

# %%
# interpolate metascore with median
df['metascore'] = df['metascore'].fillna(df['metascore'].median())
df['gross'] = df['gross'].fillna(df['gross'].median())

#come back to this

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
# director_counts = {}
# for row in df['director']:
# 	for director in row:
# 		director_counts[director] = director_counts.get(director, 0) + 1

# director_scores = pd.DataFrame(list(director_counts.items()), columns=['Director', 'Score']).sort_values(by='Score', ascending=False)
# director_scores

# %%
# new column 'Director Score' that is the sum of the scores of each director
# @lru_cache(maxsize=None)
# def get_director_score(director):
# 	return director_scores.loc[director_scores['Director'] == director, 'Score'].values[0]

# df['Director Score'] = df['director'].apply(lambda x: sum([get_director_score(director) for director in x]))
# df.drop('director', axis=1, inplace=True)

#maybe try avg instead of sum

# %%
# dataframe with all stars and the number of movies they have starred in
# star_counts = {}
# for row in df['stars']:
# 	for star in row:
# 		star_counts[star] = star_counts.get(star, 0) + 1

# star_scores = pd.DataFrame(list(star_counts.items()), columns=['Star', 'Score']).sort_values(by='Score', ascending=False)
# star_scores

# %%
# dataframe with all stars and their rating based on interpolated rating
star_ratings = {}
for row in df[['stars', 'rating']].itertuples():
	for star in row.stars:
		star_ratings[star] = star_ratings.get(star, []) + [row.rating]

star_ratings = {star: np.average(ratings) for star, ratings in star_ratings.items()}
star_ratings = pd.DataFrame(list(star_ratings.items()), columns=['Star', 'Rating']).sort_values(by='Rating', ascending=False)
star_ratings[star_ratings['Star'] == "Leonardo DiCaprio"]

# %%
# dataframe with all directors and their rating based on interpolated rating
director_ratings = {}
for row in df[['director', 'rating']].itertuples():
	for director in row.director:
		director_ratings[director] = director_ratings.get(director, []) + [row.rating]

director_ratings = {director: np.average(ratings) for director, ratings in director_ratings.items()}
director_ratings = pd.DataFrame(list(director_ratings.items()), columns=['Director', 'Rating']).sort_values(by='Rating', ascending=False)
director_ratings[director_ratings['Director'] == "Craig Moss"]

# %%
# new column 'Star Score' that is the avg of the scores of each star

@lru_cache(maxsize=None)
def get_star_score(star):
	return star_ratings.loc[star_ratings['Star'] == star, 'Rating'].values[0]

df['Star Score'] = df['stars'].apply(lambda x: np.average([get_star_score(star) for star in x]))
df.drop('stars', axis=1, inplace=True)
df['Star Score'] = df['Star Score'].fillna(df['Star Score'].median())
df

# %%
# new column 'Director Score' that is the avg of the scores of each director
@lru_cache(maxsize=None)
def get_director_score(director):
	return director_ratings.loc[director_ratings['Director'] == director, 'Rating'].values[0]

df['Director Score'] = df['director'].apply(lambda x: np.average([get_director_score(director) for director in x]))
df.drop('director', axis=1, inplace=True)
df

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
# feature_importances[:10].plot.bar()
# plt.show()
feature_importances[::-1].plot.barh()

# %%
# for length in range(1, 21):
# print(f"Top {length} features:")
# print(feature_importances[:length].index.values)

# certificate_Not Rated brought score down
# genre_Family brought score down
list = []
for length in range(1, 21):
	print(f"Top {length} features:")
	print(feature_importances[:10].index.values)
	x = df[feature_importances[:10].index.values]
	y = df['rating']

	x_train, x_test, y_train, y_test = train_test_split(x, y)

	model = XGBRegressor()
	model.fit(x_train, y_train)

	y_pred = model.predict(x_test)
	r2_score(y_test, y_pred)
	list.append(r2_score(y_test, y_pred))
# %%
# list
pd.DataFrame(list).plot.bar()
# %%
