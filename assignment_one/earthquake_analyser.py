import getopt
import sys
import json
from pathlib import Path
from earthquakes import Quake, QuakeData

# Path to the default json file
path = Path("./earthquakes.geojson")


def read_json_file(filename):
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print("Could not read the file passed in")
        exit(1)


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




