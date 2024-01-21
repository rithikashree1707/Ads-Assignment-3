#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 11:38:12 2024

@author: tamadaritikashree
"""

# Pandas for data manipulation and analysis
import pandas as pd
 # Matplotlib for plotting
import matplotlib.pyplot as plt
# KMeans for clustering
from sklearn.cluster import KMeans
# PolynomialFeatures for polynomial regression
from sklearn.preprocessing import PolynomialFeatures
# LinearRegression for linear regression
from sklearn.linear_model import LinearRegression
# Seaborn for statistical data visualization
import seaborn as sns
# Warnings library for handling warnings
import warnings

# Ignore all warnings for the purpose of this script.
warnings.filterwarnings("ignore")

#Read the CSV file, skipping the first 3 rows.
df = pd.read_csv("API_19_DS2_en_csv_v2_6300757.csv", skiprows=3)

def read_and_filter_data(file_path, indicator1, indicator2, year):
    
    """Read a CSV file from the specified path, filter the data based on given 
      indicators and year, and return a DataFrame containing the filtered data.

    Parameters:
      file_path (str): The path to the CSV file.
      indicator1 (str): The first indicator to filter the data.
      indicator2 (str): The second indicator to filter the data.
      year (int): The specific year for which the data should be filtered.
    Returns:
      pd.DataFrame: A DataFrame containing the filtered data with columns 
      'Country Name', 'indicator1', and 'indicator2'."""
      
    # Read the data
    df = pd.read_csv(file_path, skiprows=3)

    # Filter data for the specified indicators and year
    data1 = df[df["Indicator Name"] == indicator1][["Country Name", 
                                    year]].rename(columns={year: indicator1})
    data2 = df[df["Indicator Name"] == indicator2][["Country Name",
                                    year]].rename(columns={year: indicator2})

    # Merge dataframes, reset the index after merging and drop the old index
    merged_data = pd.merge(data1, data2, on="Country Name", how='outer').reset_index(drop=True)

    # Drop rows with any NaN values
    filtered_data = merged_data.dropna(how="any").reset_index(drop=True)
    
    # Return the filtered data
    return filtered_data

# Assuming file_path is the path to a CSV file 
file_path = "API_19_DS2_en_csv_v2_6300757.csv"
# Read and filter data for the year 1985
data_1985 = read_and_filter_data(file_path, 
            "CO2 emissions from liquid fuel consumption (% of total)", 
            "Urban population growth (annual %)", "1985")
# Read and filter data for the year 2005
data_2005 = read_and_filter_data(file_path, 
            "CO2 emissions from liquid fuel consumption (% of total)", 
            "Urban population growth (annual %)", "2005")


def plot_elbow(data, max_clusters=10, label=''):
    """ Plots the elbow plot for KMeans clustering.
     Parameters:
     data: The input data for KMeans clustering.
     max_clusters: The maximum number of clusters to consider.
     label: Label for the plot."""
    
    #The 'distortions' list stores
    distortions = []
    # Loop through clusters from 1 to max_clusters
    for i in range(1, max_clusters + 1):
        # Creating a KMeans clustering  with clusters using initialization
        kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
        # Fit the K-means clustering model to the given dataset
        kmeans.fit(data)
        # Append the inertia to the distortions list
        distortions.append(kmeans.inertia_)
    
    # Plot the distortion values for different numbers of clusters
    plt.plot(range(1, max_clusters + 1), distortions, marker="o", label=label)
    # Set the title of the plot
    plt.title("Elbow Plot")
    # Label the x-axis as 'Number of Clusters'
    plt.xlabel("Number of Clusters")
    # Label the y-axis as 'Distortion'
    plt.ylabel("Distortion")

# Plot elbow plots for both 2000 and 2010 data on the same figure
plt.figure(figsize=(10, 6))
# Extract relevant columns for clustering
columns_for_clustering = ["CO2 emissions from liquid fuel consumption (% of total)", 
                          "Urban population growth (annual %)"]
# Plotting the elbow method for clustering using data from 1985
plot_elbow(data_1985[columns_for_clustering], label="1985'"
# Plotting the elbow method for clustering using data from 2005
plot_elbow(data_2005[columns_for_clustering], label="2005")
# Adding a legend to the plot with labeled lines
plt.legend()
# Display the plot
plt.show()


def perform_kmeans_clustering(data, n_clusters, label=''):
    """ Perform KMeans clustering on the given data using specified 
     number of clusters.
     Parameters:
      data : Input data containing columns for clustering.
      n_clusters : Number of clusters to form.
      label : Label to include in the plot title. Default is an empty string"""
    
    # List of columns to be used for clustering
    columns_for_clustering = ["CO2 emissions from liquid fuel consumption (% of total)", 
                              "Urban population growth (annual %)"]
    # Initializing KMeans with specified parameters
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)

   
    # Use fit_predict to perform clustering and assign cluster labels 
    data["Cluster"] = kmeans.fit_predict(data[columns_for_clustering])

    # Plot a figure
    plt.figure(figsize=(8, 5))

    # Create a scatter plot to visualize the relationship
    sns.scatterplot(x="CO2 emissions from liquid fuel consumption (% of total)", 
                    y="Urban population growth (annual %)",
                    hue="Cluster", palette="viridis", data=data)

    # Scatter plot the cluster centers on the existing plot
    cluster_centers = kmeans.cluster_centers_
    # Plotting cluster centers on a scatter plot
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], 
                marker="o", s=70, c="red", label="Cluster Centers")
    # Set the title of the plot to indicate the type of clustering 
    plt.title(f'Clustering - {label}')
    # Adding a legend to the plot with labeled lines
    plt.legend()
    # Display the plot
    plt.show()

# Define the number of clusters
n_clusters = 5  

# Perform KMeans clustering for 1985 data
perform_kmeans_clustering(data_1985, n_clusters, label="1985")

# Perform KMeans clustering for 2005 data
perform_kmeans_clustering(data_2005, n_clusters, label="2005")

# Set a custom color palette for the plot
sns.set_palette("husl")

# Select three countries and the indicator
selected_countries = ["Argentina", "Sri Lanka", "Myanmar"]

# Set the indicator name
indicator_name = "CO2 emissions from liquid fuel consumption (% of total)"

# Filter the data
d_s = df[(df["Country Name"].isin(selected_countries)) & 
         (df["Indicator Name"] == indicator_name)].reset_index(drop=True)

# Melt the DataFrame
d_f = d_s.melt(id_vars=["Country Name", "Indicator Name"], 
               var_name="Year", value_name="Value")

# Filter out non-numeric values in the 'Year' column
d_f = d_f[d_f["Year"].str.isnumeric()]

# Convert 'Year' to integers
d_f["Year"] = d_f["Year"].astype(int)

# Handle NaN values by filling with the mean value
d_f["Value"].fillna(d_f["Value"].mean(), inplace=True)

# Filter data for the years between 1985 and 2020
d_f = d_f[(d_f["Year"] >= 1985) & (d_f["Year"] <= 2020)]

# Create a dictionary to store predictions for each country
predictions = {}

# Extend the range of years to include 2026
all_years_extended = list(range(1985, 2026))

# Create individual line plots for each country with a grid and unique style
for country in selected_countries:
    # Plot the figure
    plt.figure(figsize=(7, 4))
    # Plot actual data
    plt.plot(d_f[d_f["Country Name"] == country]["Year"], 
             d_f[d_f["Country Name"] == country]["Value"], 
              linestyle="-.", label=f'Actual Data', color="blue")
    
    # Prepare data for the current country
    country_data = d_f[d_f["Country Name"] == country]
    X_country = country_data["Year"]
    y_country = country_data["Value"]
    
    # Fit polynomial regression model with degree 3
    degree = 5
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X_country)
    
    model = LinearRegression()
    model.fit(X_poly, y_country)
    
    # Predict values for all years (1990 to 2025)
    X_pred = poly_features.transform(pd.DataFrame(all_years_extended, 
                                                  columns=["Year"]))
    # Use the trained machine learning model to predict values 
    forecast_values = model.predict(X_pred)
    
    # Store the predictions for the current country
    predictions[country] = forecast_values
    
    # Plot the fitted curve
    plt.plot(all_years_extended, forecast_values, label=f'Fitted Curve', 
             linestyle="-")
    
    # Plot forecast for 2025
    prediction_2025 = forecast_values[-1]
    # Plot the figure
    plt.plot(2025, prediction_2025, marker='^', 
             label=f'Prediction for 2025: {prediction_2025:.2f}', 
             color="black")
    # Set the title
    plt.title(f'{indicator_name} Forecast for {country}', fontsize=12)
    # Set the label for the x-axis  and a font size of 10
    plt.xlabel("Year", fontsize=10)
    # Set the label for the y-axis  and a font size of 10
    plt.ylabel("Kilotonns", fontsize=10)
    
    # Set x-axis limits and ticks
    plt.xlim(2000, 2030)
    # Set x-axis ticks at intervals of 5 years using Matplotlib
    plt.xticks(range(2000, 2030, 5))  
    
    # Set grid lines with a dashed linestyle and reduced transparency 
    plt.grid(True, linestyle="--", alpha=0.7)
    # Display a legend with a font size of 7 for better readability
    plt.legend(fontsize=7)
    # Display a legend with a font size of 7 for better readability
    filename = f"{indicator_name}_Forecast_{country.replace(' ', '_')}.png"
    # Save the plot with tight bounding box
    plt.savefig(filename, bbox_inches="tight")
