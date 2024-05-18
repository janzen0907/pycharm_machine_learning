import numpy as np


# Author: John Janzen(janzen0907)
# Class: COET295 - Assignment 1 Python

# TODO: Solve the filtering not working. 

# Haversine
# formula:	a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
# c = 2 ⋅ atan2( √a, √(1−a) )
# d = R ⋅ c
def calc_distance(lat1, long1, lat2, long2):
    """Will return the distance in KM between two Lat/Long Coordinates using Haversine Formula"""
    # Mean radius of the earth in metres
    radius = 6713 ** 3
    # Honestly, i'm not to sure what all these math symbols are. I googled them
    # and named them according to the reference you provided us.
    phi_one = lat1 * np.pi / 180
    phi_two = lat2 * np.pi / 180
    delta_phi = (lat2 - lat1) * np.pi / 180
    delta_lamda = (long2 - long1) * np.pi / 180

    a = np.sin(delta_phi / 2) * np.sin(delta_phi / 2) + np.cos(phi_one) * np.cos(phi_two) * np.sin(
        delta_lamda / 2) * np.sin(delta_lamda / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = radius * c
    # print(f"In calc distance {d}")
    return d


class QuakeData:
    """Class to represent data regarding Earth Quakes"""

    def __init__(self, geojson):
        # Array to hold valid EarthQuaks
        self.quake_array = []

        # Filter attributes
        self.filter_latitude = None
        self.filter_longiture = None
        self.filter_distance = None
        self.filter_magnitude = None
        self.filter_felt = None
        self.filter_time = None
        self.filter_significance = None

        # Get the features from the dict
        features = geojson.get('features', [])

        # Var to hold the datatype structure of our data
        dtype = [
            ('quake', object),
            ('magnitude', float),
            ('felt', np.int32),
            ('significance', np.int32),
            ('lat', float),
            ('long', float)
        ]

        # Var to hold the data
        data = []

        # Loop through the features
        for feature in features:
            # Check if the type is a feature
            if feature.get('type') == 'Feature':
                properties = feature.get('properties', {})
                geometry = feature.get('geometry', {})

                # Ensure that properties contains the proper fields
                if 'mag' in properties and 'time' in properties and 'felt' in properties and 'sig' in properties and 'type' in properties:
                    # Check that point does not exist
                    if geometry.get('type') == 'Point' and 'coordinates' in geometry and len(
                            geometry['coordinates']) == 3:
                        # Get the data and build the array
                        quake = feature
                        magnitude = properties['mag']
                        felt = properties['felt'] if properties['felt'] is not None else 0
                        signifigance = properties['sig']
                        lat, long = geometry['coordinates'][:2]

                        # Add the data to the list
                        data.append((quake, magnitude, felt, signifigance, lat, long))

        # If the data is correct. Convert it to a nump array
        if data:
            self.quake_array = np.array(data, dtype=dtype)

    def set_location_filter(self, latitude, longitude, distance):
        """Returns Quakes within the distance the distance of the latitude and longitude being passed in"""
        self.filter_latitude = latitude
        self.filter_longiture = longitude
        self.filter_distance = distance

    def set_property_filter(self, magnitude=None, felt=None, significance=None):
        """Return only quakes whose magnitude, felt, or significance properties are at least the values supplied"""
        # Check that one property was passed in
        if magnitude is None and felt is None and significance is None:
            raise ValueError("Please supply at least one parameter")

        self.filter_magnitude = magnitude
        self.filter_felt = felt
        self.filter_significance = significance

    def clear_filter(self, magnitude=None, felt=None, significance=None, lat=None, long=None, distance=None):
        """Will clear all the filters"""
        self.filter_distance = None
        self.filter_felt = None
        self.filter_latitude = None
        self.filter_longiture = None
        self.filter_magnitude = None
        self.filter_significance = None

    def get_filtered_array(self):
        """Returns a numpy array of only quakes that meet the passed in critera to the filters"""
        # Check if there is quake data in th quake array
        if self.quake_array is None:
            return None

        quakes_filtered = self.quake_array

        # Filters for the location
        if self.filter_latitude is not None and self.filter_longiture is not None and self.filter_distance is not None:
            quakes_filtered = np.array([
                quake for quake in quakes_filtered
                if calc_distance(self.filter_latitude, self.filter_longiture, quake['lat'],
                                 quake['long']) <= self.filter_distance
            ])

        # Filters for the properties
        if self.filter_magnitude is not None or self.filter_felt is not None or self.filter_significance is not None:
            quakes_filtered = np.array([
                quake for quake in quakes_filtered
                if (self.filter_magnitude is None or quake['magnitude'] >= self.filter_magnitude) and
                   (self.filter_felt is None or quake['felt'] >= self.filter_felt) and
                   (self.filter_significance is None or quake['significance'] >= self.filter_significance)
            ])

            # for quake in self.quake_array:
            #     quake_lat = quake['lat']
            #     quake_long = quake['long']
            #
            #     distance = calc_distance(self.filter_latitude, self.filter_longiture, quake_lat, quake_long)
            #     if distance is None:
            #         distance = 0
            #
            #     if distance <= self.filter_distance:
            #         quakes_filtered.append(quake)
            return quakes_filtered

        # Filters for the properties.
        if self.filter_magnitude is not None or self.filter_felt is not None or self.filter_significance is not None:
            quakes_filtered = []
            for quake in self.quake_array:
                # Using / to break this wonderful if statement onto multipl lines for readability
                if self.filter_magnitude is None or quake['magnitude'] >= self.filter_magnitude and \
                        self.filter_felt is None or quake['felt'] >= self.filter_felt and \
                        self.filter_significance is None or quake['significance'] >= self.filter_significance:
                    return np.array(quakes_filtered)

        # No location was set, return the original array
        return self.quake_array

    def get_filtered_list(self):
        """Return a list of Quake objects containing the quakes that met the above filters"""
        # Call the function to get a filtered array
        filtered_array = self.get_filtered_array()

        # Ensure it contains data
        if filtered_array is not None:
            filtered_list = []
            for quake in filtered_array:
                # Loop through the array and add the quaakes to the list
                filtered_list.append(quake['quake'])
            return filtered_list
        return None


class Quake:
    """Contains data about a quake"""

    def __init__(self, magnitude, time, felt, significance, q_type, coords):
        self.magnitude = magnitude
        self.time = time
        self.felt = felt
        self.significance = significance
        self.q_type = q_type
        self.lat = coords[0]
        self.long = coords[1]

    def __str__(self):
        """Returns a constructed string to represent a Quake Object"""
        return f"{self.magnitude} Magnitude Earthquake, {self.significance} Significance, felt by {self.felt} people in ({self.lat}, {self.long})"

    def get_distance_from(self, latitude, longitude, distance):
        """Return the number of KM the quake is away from the passed in point"""
        # Calling my previously made function for this, I don't think anything really changes so I will call the method
        calc_distance(self.lat, self.long, latitude, longitude)

# Testing Quake string
# test_quake = Quake(1,5.0,10,5,1,(10,100.5))
# print(test_quake)
# print(test_quake)
