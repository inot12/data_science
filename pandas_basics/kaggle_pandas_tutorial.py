#! ~/python-environments/3.8.5/bin/python3
"""
Created on Feb 18, 2022

@author:inot
"""

import numpy as np
import pandas as pd

# pd.set_option('max_rows', 5)

# ============================================================================
# Creating, Reading and Writing
# ============================================================================
# DataFrame: A table containing an array of individual entries, where each has
# a certain value.
yes_no_table = pd.DataFrame({'Yes': [50, 21], 'No': [131, 2]})
# Even if there is only one data element in a column, it must be specified in
# a list.
fruits = pd.DataFrame({'Apples': [30], 'Bananas': [21]})
strings_table = pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'],
                              'Sue': ['Pretty good.', 'Bland.']})
index_row_labels = pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'],
                                 'Sue': ['Pretty good.', 'Bland.']},
                                index=['Product A', 'Product B'])

# Series: A sequence of data values. In essence, it is a single column of a
# DataFrame.
series_list = pd.Series([1, 2, 3, 4, 5])
named_series = pd.Series([30, 35, 40],
                         index=['2015 Sales', '2016 Sales', '2017 Sales'],
                         name='Product A')

# Reading data files
wine_reviews = pd.read_csv("./data/winemag-data-130k-v2.csv.zip")
wine_reviews_shape = wine_reviews.shape  # Check the size of the DataFrame.
# Grab the 1st 5 rows of DataFrame wine_reviews.
wine_reviews_first_five = wine_reviews.head()
# Use the built-in index of the csv file with index_col, instead creating
# a new index that will duplicate the built-in index.
builtin_index_wine_reviews = pd.read_csv(
    "./data/winemag-data-130k-v2.csv.zip", index_col=0)
indexed_wine_reviews = builtin_index_wine_reviews.head()
# Save a DataFrame to disk as a csv file.
indexed_wine_reviews.to_csv('indexed_wine_reviews.csv')

# ============================================================================
# Indexing, Selecting and Assigning
# ============================================================================
reviews = builtin_index_wine_reviews
# Access properties as an attribute. In thise case we access the property
# country in the reviews DataFrame.
countries = reviews.country
# Same as reviews.country.
country = reviews['country']
# Access a single value from a Series in DataFrame.
first_country = reviews['country'][0]
first_description = reviews.description.iloc[0]

# Pandas preferred accessor operators: loc and iloc
# Index-based selection with iloc based on numerical position in the data
# .iloc[0: 10] selects entries 0, ..., 9
first_row = reviews.iloc[0]
first_column = reviews.iloc[:, 0]
three_row_column = reviews.iloc[:3, 0]  # rows 0, 1, 2
selected_range_column = reviews.iloc[5:8, 0]  # 5th to 7th rows
selected_rows_column = reviews.iloc[[0, 8, 12], 0]  # rows 0, 8 and 12
sample_reviews = reviews.iloc[[1, 2, 3, 5, 8], :]
last_country = reviews['country'].iloc[-1]
last_five_rows = reviews.iloc[-5:]
first_ten_descriptions1 = reviews.description[:10]
first_ten_descriptions2 = reviews.description.iloc[:10]
first_ten_descriptions3 = reviews.loc[:9, 'description']
first_hundred_select_columns = reviews.loc[:99, ['country', 'variety']]
# Label-based selection with loc based on data index value
# .loc[0:10] selects entries 0, ..., 10
first_column_with_loc = reviews.loc[:, 'country']
all_rows_in_select_columns = \
    reviews.loc[:, ['taster_name', 'taster_twitter_handle', 'points']]

# Manipulating the index
changed_index = reviews.set_index('title')

# Conditional selection
italian_wine = reviews.country == 'Italy'
show_italian_wine = reviews.loc[reviews.country == 'Italy']
# Joined conditional selection
show_good_italian_wine = reviews.loc[
    (reviews.country == 'Italy') & (reviews.points >= 90)]  # logical and
show_any_italian_or_good_wine = reviews.loc[
    (reviews.country == 'Italy') | (reviews.points >= 90)]  # logical or
# Select data whose value is in a list of values.
wines_only_from_italy_or_france = reviews.loc[
    reviews.country.isin(['Italy', 'France'])]
top_oceania_wines1 = reviews.loc[
    (reviews.points >= 95) & (
        (reviews.country == 'Australia') | (reviews.country == 'New Zealand'))]
top_oceania_wines2 = reviews.loc[
    (reviews.points >= 95) &
    (reviews.country.isin(['Australia', 'New Zealand']))]
# Select data which are empty.
no_price_tag = reviews.loc[reviews.price.isnull()]
# Select data which are not empty.
price_tag_exists = reviews.loc[reviews.price.notnull()]

# Assigning data
everyone_is_a_critic = reviews['critic'] = 'everyone'
index_backwards = reviews['index_backwards'] = range(len(reviews), 0, -1)

# ============================================================================
# Summary functions and maps
# ============================================================================
# "Summary functions restructure data in some useful way.
# describe() is a type-aware method that generates a high-level summary of the
# attributes of the given column.
points_summary = reviews.points.describe()
taster_name_summary = reviews.taster_name.describe()
points_mean = reviews.points.mean()  # mean value
points_median = reviews.points.median()
taster_name_unique = reviews.taster_name.unique()  # a list of unique values
# show a list of unique values and how often they occur in the dataset
taster_name_value_counts = reviews.taster_name.value_counts()

# Maps are functions that map one set of values to another set of values.
# map() and apply() don't modify the original data, but return new transformed
# Series and DataFrames.
wine_scores_minus_mean_value1 = reviews.points.map(lambda p: p - points_mean)
# Select the name of the first wine reviewed from each winery.
first_reviewed_wine_from_each_winery = \
    reviews.groupby('winery').apply(lambda df: df.title.iloc[0])


def remean_points(row):
    row.points = row.points - points_mean
    return row


# Set axis='index' to transform each column.
remean_each_row_in_dataframe = reviews.apply(remean_points, axis='columns')
# The following operators are FASTER than map() and apply(), but they lack
# flexibilty in doing more advanced things, like applying conditional logic.
# Alternative to map().
wine_scores_minus_mean_value2 = reviews.points - points_mean
# Similar to the operation with one Series and one value, we can operate on
# two Series of equal length.
combine_country_and_region = reviews.country + ' - ' + reviews.region_1
# Title of wine with the highest point to price ratio.
bargain_idx = (reviews.points / reviews.price).idxmax()
bargain_wine = reviews.loc[bargain_idx, 'title']
# Count how many times 'tropical' and 'fruity' appear in description column.
n_tropical = reviews.description.map(lambda desc: "tropical" in desc).sum()
n_fruity = reviews.description.map(lambda desc: "fruity" in desc).sum()
descriptor_counts = pd.Series(
    [n_tropical, n_fruity], index=['tropical', 'fruity'])


# Translate the rating system ranging from 80 to 100 points into star ratings.
# Score >= 95 = 3 stars, 85 <= Score < 95 = 2 stars, Score < 85 = 1 star.
# Any wine from Canada = 3 stars.
def stars(row):
    if row.country == 'Canada':
        return 3
    if row.points >= 95:
        return 3
    if row.points >= 85:
        return 2
    return 1


star_ratings = reviews.apply(stars, axis='columns')

# ============================================================================
# Grouping and sorting
# ============================================================================
# In this case, groupby() creates a group of reviews which gave the same
# point score and then count() counts how many reviews of the same score were
# issued. value_counts() is a shortcut for this groupby() with count() command.
points_count = reviews.groupby('points').points.count()
# Get the cheapest wine in each point value category.
cheapest_wine_by_point_score = reviews.groupby('points').price.min()
# Group by more than one column. Pick the best wine by country and province.
best_wine_by_country_and_province = \
    reviews.groupby(['country', 'province']).apply(
        lambda df: df.loc[df.points.idxmax()])
# agg() enables you to run a lot of different functions on your DataFrame.
statistical_summary = reviews.groupby(['country']).price.agg([len, min, max])
# Multi-index grouping.
countries_reviewed = reviews.groupby(
    ['country', 'province']).description.agg([len])
# The most used multi-index method is reset_index() that converts the
# multi-index to a regular index. This implies that multi-index is rarely used.
reset_index = countries_reviewed.reset_index()

# Sorting by a desired parameter is accomplished with sort_values(). The
# default sorting order is ascending.
sort_reviewed_countries_with_reset_index_by_len = \
    reset_index.sort_values(by='len')
# Sort in descending order.
descending_sort = reset_index.sort_values(by='len', ascending=False)
# Sort by index.
sort_by_index = countries_reviewed.sort_index()
# Sort by more columns at the same time.
multi_column_sort = reset_index.sort_values(by=['country', 'len'])
# Crate a multi-index Series of country and variety and sort descending based
# on wine count.
country_variety_counts = reviews.groupby(
    ['country', 'variety']).size().sort_values(ascending=False)

# ============================================================================
# Data Types and Missing Values
# ============================================================================
# The data type for a column in a DataFrame or a Series is known as the dtype.
# Columns consisting entiery of strings are given the object type.
# You can use the dtype property to grab the type of a specific column.
price_data_type = reviews.price.dtype
# Return dtype of every column in a Data Frame.
all_columns_data_type = reviews.dtypes
# Convert a column type to another with astype()
points_from_int64_to_float64 = reviews.points.astype('float64')
# DataFrame index dtype
index_dtype = reviews.index.dtype

# Missing data has a value NaN of float64 dtype. NaN entries can be selected
# with pd.isnull().
NaN_entries = reviews[pd.isnull(reviews.country)]
# Replace missing values with fillna(). The backfill strategy is to replace
# each missing value with the first appearing non-null value that comes after
# the missing value.
NaN_to_unknown = reviews.region_2.fillna("Unknown")
# Replace any value with the replace() method. Useful for replacing missing
# data that has some sentinel value like "Unknown", "Undisclosed", "Invalid".
update_reviewer_info = reviews.taster_twitter_handle.replace(
    "@kerinokeefe", "@kerino")
number_of_missing_prices = len(reviews[pd.isnull(reviews.price)])

# ============================================================================
# Renaming and Combining
# ============================================================================
# Change index names or column names with rename().
rename_points_to_score = reviews.rename(columns={'points': 'score'})
rename_elements_of_the_index = reviews.rename(
    index={0: 'firstEntry', 1: 'secondEntry'})
set_row_index_name_and_column_index_name = \
    reviews.rename_axis("wines", axis='rows').rename_axis(
        "fields", axis='columns')

# Pandas has three core methods for combining different DataFrames and Series.
# These are concat(), join() and merge(). Since join() can do mostly everything
# that merge() does and is simpler to use, merge() is less frequently used.
# concat() is used to combine data in different DataFrames or Series that have
# the same columns.
# join() is used to combine different DataFrames that have an index in common.


def main():
    print(yes_no_table)
    print()
    print(fruits)
    print()
    print(strings_table)
    print()
    print(index_row_labels)
    print()
    print(series_list)
    print()
    print(named_series)
    print()
    print(wine_reviews)
    print()
    print(wine_reviews_shape)
    print()
    print(wine_reviews_first_five)
    print()
    print(builtin_index_wine_reviews)
    print()
    print(indexed_wine_reviews)
    print()
    print(countries)
    print()
    print(country)
    print()
    print(first_country)
    print()
    print(first_description)
    print()
    print(first_row)
    print()
    print(first_column)
    print()
    print(three_row_column)
    print()
    print(selected_range_column)
    print()
    print(selected_rows_column)
    print()
    print(sample_reviews)
    print()
    print(last_country)
    print()
    print(last_five_rows)
    print()
    print(first_ten_descriptions1)
    print()
    print(first_ten_descriptions2)
    print()
    print(first_ten_descriptions3)
    print()
    print(first_hundred_select_columns)
    print()
    print(first_column_with_loc)
    print()
    print(all_rows_in_select_columns)
    print()
    print(changed_index)
    print()
    print(italian_wine)
    print()
    print(show_italian_wine)
    print()
    print(show_good_italian_wine)
    print()
    print(show_any_italian_or_good_wine)
    print()
    print(wines_only_from_italy_or_france)
    print()
    print(top_oceania_wines1)
    print()
    print(top_oceania_wines2)
    print()
    print(no_price_tag)
    print()
    print(price_tag_exists)
    print()
    print(everyone_is_a_critic)
    print()
    print(reviews['critic'])
    print()
    print(index_backwards)
    print()
    print(reviews['index_backwards'])
    print()
    print(points_summary)
    print()
    print(taster_name_summary)
    print()
    print(points_mean)
    print()
    print(points_median)
    print()
    print(taster_name_unique)
    print()
    print(taster_name_value_counts)
    print()
    print(wine_scores_minus_mean_value1)
    print()
    print(remean_each_row_in_dataframe)
    print()
    print(wine_scores_minus_mean_value2)
    print()
    print(combine_country_and_region)
    print()
    print(bargain_idx)
    print()
    print(bargain_wine)
    print()
    print(n_tropical)
    print()
    print(n_fruity)
    print()
    print(descriptor_counts)
    print()
    print(star_ratings)
    print()
    print(points_count)
    print()
    print(cheapest_wine_by_point_score)
    print()
    print(first_reviewed_wine_from_each_winery)
    print()
    print(best_wine_by_country_and_province)
    print()
    print(statistical_summary)
    print()
    print(countries_reviewed)
    print()
    print(reset_index)
    print()
    print(sort_reviewed_countries_with_reset_index_by_len)
    print()
    print(descending_sort)
    print()
    print(sort_by_index)
    print()
    print(multi_column_sort)
    print()
    print(country_variety_counts)
    print()
    print(price_data_type)
    print()
    print(all_columns_data_type)
    print()
    print(points_from_int64_to_float64)
    print()
    print(index_dtype)
    print()
    print(NaN_entries)
    print()
    print(NaN_to_unknown)
    print()
    print(update_reviewer_info)
    print()
    print(number_of_missing_prices)
    print()
    print(rename_points_to_score)
    print()
    print(rename_elements_of_the_index)
    print()
    print(set_row_index_name_and_column_index_name)


if __name__ == "__main__":
    main()
