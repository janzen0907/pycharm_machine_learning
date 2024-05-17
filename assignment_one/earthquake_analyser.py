import getopt
import sys
import json
from pathlib import Path
from earthquakes import Quake, QuakeData

# NOTE: for me it is D:\pycharm\assignment_one\earthquake_analyser.py
# TODO: Test the filters and continue the rest of the choices,
# reading in the file is working now

path = Path("D:/pycharm/assignment_one/earthquakes.geojson")
print(path)
# entries = path.read_text()
# print(entries)
# if len(sys.argv) > 1:
#     path == sys.argv[1]
#     earthquakes = json.loads(sys.argv[1].read_text())
# elif len(sys.argv) == 1:
#     earthquakes = json.loads(path)

# if path.exists():
#     quake_data = json.loads(path.read_text())
#     print(quake_data)
# else:
#     print(f"Not working. {path}")


# def read_json_file(filename):
#     try:
#         with open(filename, 'r') as file:
#             earthquakes = json.load(file)
#             print("File opened successfully")
#     except FileNotFoundError:
#         print("Could not read the file passed in")
#         exit(1)
# -71.4113,
#                     -16.8393,
#                     118.01

def display_menu():
    """Display a menu for the user to choose what operation they would like to perform"""
    print("Please select an option from the following list\n")
    print("1: Set location filter\n")
    print("2: Set property filter\n")
    print("3: Clear filters\n")
    print("4: Display Quakes\n")
    print("5: Clear Filters\n")
    print("6: Display Exceptional Quakes\n")
    print("7: Display Magnitude Stats\n")
    print("8: Plot Quake Map\n")
    print("9: Plot Magnitude Chart\n")
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


# Check the supplied arguments. If no arguments are passed in we will read it in from earthquakes.geojson
if len(sys.argv) == 2:  # Check if exactly one command-line argument is provided
    path = Path(sys.argv[1])
    # print(sys.argv[1])
    # print(path)
    earthquakes = json.loads(path.read_text())
    quake_data = QuakeData(earthquakes)
    # print(quake_data)
elif len(sys.argv) == 1:  # If no command-line arguments provided, use default file
    # print(f"In the elif: {path} {path.read_text()}")
    try:
        earthquakes = json.loads(path.read_text())
        quake_data = QuakeData(earthquakes)
    except json.JSONDecodeError as je:
        # Getting a jsonDecode error now all of a sudden. figure this out.
        print(f"error {je}")
    # print(earthquakes)
    # print(quake_data)
else:
    print("Usage: python script.py [json_file]")
    sys.exit(1)

# filename = sys.argv[1] if len(sys.argv) > 1 else str(path)
#
# earthquakes = read_json_file(filename)
# print(earthquakes)
# quake_data = QuakeData(earthquakes)


# Loop to keep interacting with the user
while True:
    display_menu()
    choice = input("Enter your choice")

# Ensure that the input is a int
    try:
        choice = int(choice)
    except ValueError:
        print("Please enter a valid number (0-9)")
        continue

# Logic for the different options
    if choice == 1:  # Get the values from the get function
        filter_lat, filter_long, filter_distance = get_lat_long_distance()
        # Ensure none of the values are None
        if filter_lat is not None and filter_long is not None and filter_distance is not None:
            quake_data.set_location_filter(filter_lat, filter_long, filter_distance)
        print(f"Filtering by Latitude: {filter_lat}, Longitude: {filter_long}, and Distance: {filter_distance}")

    elif choice == 2:
        filter_sig, filter_felt, filter_mag = get_sig_felt_mag()
        quake_data.set_property_filter(filter_mag, filter_felt, filter_sig)
        print(f"Filtering by Magnitude: {filter_mag}, Felt: {filter_felt}, and Significance: {filter_sig}")

    elif choice == 3:
        quake_data.clear_filter()

    elif choice == 4:
        print(quake_data)
        print(quake_data.get_filtered_list())

    elif choice == 5:
        pass

    elif choice == 6:
        pass

    elif choice == 7:
        pass

    elif choice == 8:
        pass

    elif choice == 9:
        pass

    elif choice == 0:
        quit(1)
