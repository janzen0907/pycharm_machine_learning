import getopt
import os
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from assignment_one import earthquakes
from earthquakes import Quake, QuakeData

# Path to the default json file
path = Path("./earthquakes.geojson")

# print(f"Current working directory: {os.getcwd()}")
# print(path)
# print(path.read_text())


def display_menu():
    """Display a menu for the user to choose what operation they would like to perform"""
    print("Please select an option from the following list\n")
    print("1: Set location filter\n")
    print("2: Set property filter\n")
    print("3: Clear filters\n")
    print("4: Display Quakes\n")
    print("5: Display Exceptional Quakes\n")
    print("6: Display Magnitude Stats\n")
    print("7: Plot Quake Map\n")
    print("8: Plot Magnitude Chart\n")
    print("0: Quit the program")

def get_lat_long_distance():
    """Function to promt the user for the lat, long and distance values."""
    filter_lat = input("Enter Latitude: ")
    filter_long = input("Enter Longitude: ")
    filter_distance = input("Enter Distance: ")
    try:
        filter_lat = float(filter_lat)
        filter_long = float(filter_long)
        filter_distance = float(filter_distance)
    except ValueError:
        print("Values must be numbers")
        return None, None, None
    return filter_lat, filter_long, filter_distance

def get_sig_felt_mag():
    """Function to promt the user for the significance, felt and longitude values."""
    filter_sig = input("Enter Minimum Significance: ")
    filter_felt = input("Enter Minimum Felt: ")
    filter_mag = input("Enter Minimum Magnitude: ")

    if filter_sig is not None and filter_felt is not None and filter_mag is not None:
        filter_sig = float(filter_sig)
        filter_felt = float(filter_felt)
        filter_mag = float(filter_mag)
    else:
        print("No filter has been set for one of the properties, using default value of None")
        return filter_sig, filter_felt, filter_mag

    return filter_sig, filter_felt, filter_mag

def get_exceptional_quakes(quake_data):
    """Function that will return the quakes that are greater than one standard deviation above the median quake magnitude"""
    # filtered_quakes = quake_data.get_filtered_list()
    filtered_quakes = quake_data.get_filtered_array()
    # If filterer quakes is empty return an empty list
    if len(filtered_quakes) == 0 or filtered_quakes is None:
        return []

    magnitudes = filtered_quakes['magnitude']
    # Get the median magnitudes for the list
    median_mag = np.median(magnitudes)
    # Calculate the standard deviation
    std_dev = np.std(magnitudes)
    threshold = median_mag + std_dev

    exep_quakes = []
    # Loop through the quakes and filter out all that don't match the criteria
    for quake in filtered_quakes:
        if quake['magnitude'] > threshold:
            exep_quakes.append(**quake['quake'])

    return exep_quakes


def display_mag_stats(quake_data):
    """Function that will display the mode, median, standard deviation and the mean of the filtered quakes"""
    filtered_quakes = quake_data.get_filtered_array()
    if len(filtered_quakes) == 0 or filtered_quakes is None:
        print("No quakes available to show")
        return

    magnitudes = filtered_quakes['magnitude']
    mean_mag = np.mean(magnitudes)
    std_dev = np.std(magnitudes)
    median_mag = np.median(magnitudes)

    # Get the mode of the magnitudes
    rounded_mag = np.floor(magnitudes)
    # Get the values and how many of each there are
    values, counts = np.unique(magnitudes, return_counts=True)
    mode_index = np.argmax(counts)
    mode_mag = values[mode_index]

    print(f"Mean Magnitude: {mean_mag: .2f}")
    print(f"Median Magnitude: {median_mag: .2f}")
    print(f"Standard Deviation: {std_dev: .2f}")
    print(f"Mode: {mode_mag}")


def plot_quake_map(quake_data):
    """Display a scatter map of teh filtered quakes"""
    filtered_quakes = quake_data.get_filtered_array()
    if filtered_quakes is None or len(filtered_quakes) == 0:
        print("No quakes available to show")
        return

    lats = filtered_quakes['lat']
    longs = filtered_quakes['long']
    magnitudes = filtered_quakes['magnitude']

    plt.figure(figsize=(10, 6))
    plt.scatter(longs, lats, s=magnitudes**2, alpha=0.5)
    plt.colorbar(label='Magnitude')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Scatter map for Filtered Quakes')
    plt.show()


def plot_magnitude_chart(quake_data):
    """Display a bar chart of how many quakes occured amonst the filtered quakes"""
    filtered_quakes = quake_data.get_filtered_array()
    if filtered_quakes is None or len(filtered_quakes) == 0:
        print("No quakes available to show")
        return

    magnitudes = np.floor(filtered_quakes['magnitude'].astype(int))
    unique_mags, counts = np.unique(magnitudes, return_counts=True)

    plt.figure(figsize=(10, 6))
    plt.bar(unique_mags, counts, width=0.5, align='center')
    plt.xlabel('Magnitude')
    plt.ylabel('Number of Quakes')
    plt.title('Quakes by Magnitude')
    plt.xticks(unique_mags)
    plt.show()



def main():
    # Check the supplied arguments. If no arguments are passed in we will read it in from earthquakes.geojson
    if len(sys.argv) != 1:
        # Read in the default file
        earthquakes = json.loads(path.read_text())
        quake_data = QuakeData(earthquakes)

    # If an arg was passed in read in the supplied script
    elif len(sys.argv) == 1:
        # print the name of the script
        print(f"Reading in {sys.argv[0]}")
        # Variable for the script name
        script_name = sys.argv[0]
        # Path for the script
        script_path = Path(f"./{script_name}")
        earthquakes = json.loads(script_path.read_text())
        quake_data = QuakeData(earthquakes)




