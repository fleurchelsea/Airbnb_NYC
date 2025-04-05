import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("AB_NYC_2019.csv")  # Ensure this CSV is in the same directory

# Clean data
df.dropna(subset=['reviews_per_month'], inplace=True)
df = df[df['price'] > 0]  # Remove listings with zero price

# Outliers: remove outliers for better visualization
df = df[df['price'] < 1000]

# Section 1: Distribution by Neighborhood Group
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='neighbourhood_group', order=df['neighbourhood_group'].value_counts().index, palette='Set2')
plt.title('Number of Listings by NYC Neighborhood Group')
plt.xlabel('Neighbourhood Group')
plt.ylabel('Number of Listings')
plt.tight_layout()
plt.savefig('figure_1_neighbourhood_group_distribution.png')
plt.clf()

# Focus analysis on Manhattan and Brooklyn
df_focus = df[df['neighbourhood_group'].isin(['Manhattan', 'Brooklyn'])]

# Section 2: Log Price vs Reviews by Neighborhood
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
for ax, name in zip(axes, ['Brooklyn', 'Manhattan']):
    subset = df_focus[df_focus['neighbourhood_group'] == name]
    sns.scatterplot(x='number_of_reviews', y='price', data=subset, ax=ax, alpha=0.3)
    sns.regplot(x='number_of_reviews', y=np.log(subset['price']), scatter=False, ax=ax, color='blue')
    ax.set_title(name)
    ax.set_xlabel('Number of Accommodation Reviews')
    ax.set_ylabel('Logarithm of Accommodation Price in USD')
plt.suptitle('Logarithm of Airbnb Prices in USD by Review and Neighbourhood')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('figure_2_log_price_vs_reviews.png')
plt.clf()

# Section 3: Mean Price by Room Type (Manhattan)
manhattan = df[df['neighbourhood_group'] == 'Manhattan']
room_type_avg = manhattan.groupby('room_type')['price'].mean().reset_index()

plt.figure(figsize=(8, 6))
sns.barplot(x='room_type', y='price', data=room_type_avg, palette='pastel')
plt.title('Mean Price of Airbnb Listings in Manhattan by Room Type')
plt.ylabel('Mean of Airbnb Prices (USD)')
plt.xlabel('Room Type')
plt.tight_layout()
plt.savefig('figure_3_mean_price_by_room_type.png')
plt.clf()

# Section 4: Prices by Neighborhood (Manhattan only)
neigh_avg = manhattan.groupby('neighbourhood').agg({'price': 'mean', 'latitude': 'mean', 'longitude': 'mean'}).reset_index()

plt.figure(figsize=(10, 8))
sns.scatterplot(data=neigh_avg, x='longitude', y='latitude', size='price', hue='neighbourhood', legend=False, sizes=(20, 400))
plt.title('Airbnb Listing Distribution and Average Prices by Manhattan Neighborhood in 2019')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.savefig('figure_4_neighborhood_price_distribution.png')
plt.clf()

# Section 5: Minimum Nights vs Price (Manhattan)
manhattan_filtered = manhattan[manhattan['minimum_nights'] <= 60]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='minimum_nights', y='price', data=manhattan_filtered, alpha=0.3)
sns.regplot(x='minimum_nights', y=np.log(manhattan_filtered['price']), scatter=False, color='blue')
plt.yscale('log')
plt.xlabel('Minimum nights of stay')
plt.ylabel('Logarithm of price per night in USD')
plt.title('Relationship Between Logarithmic Price per Night and Minimum Stay Requirement for Airbnbs in Manhattan')
plt.tight_layout()
plt.savefig('figure_5_minimum_nights_vs_price.png')
plt.clf()

print("Analysis complete. Figures saved to current directory.")
