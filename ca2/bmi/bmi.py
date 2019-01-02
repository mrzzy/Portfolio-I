#
# bmi.py
# Portfolio I
# Data Science with BMI data
#

# Add imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import shutil
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.plotting import plot_decision_regions

# Plot a histogram with the given data and bin_size and colpor
def plot_hist(x, bin_size, color, label):
    bins = np.arange(min(x), max(x) + bin_size, bin_size)
    plt.hist(x, bins=bins, label=label, color=color)

# Save the current plot with the given filename
def save_plot(name):
    plt_path = os.path.join("plots", name)
    plt.savefig(plt_path)
    plt.clf()

# Compute BMI for the given weights and heights
def compute_bmi(weight, height):
    return weight / ((height / 100) ** 2)

def print_stats(x, label):
    print("{} > Mean: {:.2f} Median: {:.2f} Std Dev: {:.2f}".format(
          label, np.mean(x), np.median(x), np.std(x)))

if __name__ == "__main__":
    # Load data as dataframe
    bmi_df = pd.read_csv("bmi.csv")
    n_data =  len(bmi_df)
    regions = ["Asia", "Africa", "Europe"]
    headers = list(bmi_df.columns.values)[1:]
    weight_headers = [h for h in headers if "Wt" in h]
    height_headers = [h for h in headers if "Ht" in h]
    
    # Compute BMI
    weight_mat = np.asarray([bmi_df[h].values for h in weight_headers])
    height_mat = np.asarray([bmi_df[h].values for h in height_headers])
    bmi_mat = compute_bmi(weight_mat, height_mat)
    
    ## 1. PLOTS
    # Create directory to output plot if does not already exist
    shutil.rmtree("plots", ignore_errors=True)
    os.mkdir("plots")
    
    # Histogram Plot of the heights distribution
    plt.title("Height distribution of Asia, Africa and Europe")
    plt.ylabel("Height (cm)")
    plt.xlim((150, 200))
    colors = ["darksalmon", "coral", "orange"]
    for i, header in enumerate(height_headers):
        heights = bmi_df[header]
        plot_hist(heights, bin_size=4, color=colors[i], label=header)
    plt.legend()
    save_plot("height.png")

    # Histogram Plot of the weights distribution
    plt.title("Weight distribution of Asia, Africa and Europe")
    plt.ylabel("Weight (cm)")
    colors = ["darksalmon", "coral", "orange"]
    for i, header in enumerate(weight_headers):
        weights = bmi_df[header]
        plot_hist(weights, bin_size=4, color=colors[i], label=header)
    plt.legend()
    save_plot("weight.png")
    
    # Scatterplot of the BMI of each region
    plt.title("BMI distribution of Asia, Africa and Europe")
    plt.ylabel("BMI")
    plt.ylim((15, 30))
    for i, region in enumerate(regions):
        bmi_vals = sorted(bmi_mat[i])
        plt.plot(range(n_data), bmi_vals, "x", color=colors[i], label=region)
    plt.legend()
    save_plot("bmi.png")

    ## 2. Statistics
    for header in headers:
        values = bmi_df[header]
        print_stats(values, header)
    
    for i, region in enumerate(regions):
        values = bmi_mat[i]
        print_stats(values, "{} BMI".format(region))
    
    ## 3. Machine Learning
    # Construct dataset
    labels = []
    features = []
    for i, region in enumerate(regions):
        for bmi in bmi_mat[i]:
            labels.append(i)
            features.append(bmi)
             
    labels = np.asarray(labels)
    features = np.reshape(features, (-1, 1))
    
    # Train KNN model on data
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(features, labels)
    print(features.shape, labels.shape)

    # Plot decision boardaries of model
    plt.title("KNN Decision Boardaries")
    plot_decision_regions(X=features, y=labels, clf=model, legend=2)
    plt.xlim((0, 600))
    plt.show()
    
    # Interractive client to predict region given height and weight
    while True:
        height = float(input("Enter height: "))
        weight = float(input("Enter weight: "))
        
        bmi = compute_bmi(weight, height)
        test_features = np.reshape(bmi, (1, 1))
        predictions = model.predict(test_features)
        print("Predicted region: ", regions[predictions[0]])
