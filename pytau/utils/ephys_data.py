"""
Define helper class for dealing with blech_clust associated data transactions
"""

#  ______       _                      _____        _
# |  ____|     | |                    |  __ \      | |
# | |__   _ __ | |__  _   _ ___ ______| |  | | __ _| |_ __ _
# |  __| | '_ \| '_ \| | | / __|______| |  | |/ _` | __/ _` |
# | |____| |_) | | | | |_| \__ \      | |__| | (_| | || (_| |
# |______| .__/|_| |_|\__, |___/      |_____/ \__,_|\__\__,_|
#        | |           __/ |
#        |_|          |___/


import glob
import json
import os

import easygui
import numpy as np
import tables


class EphysData:
    """Class to streamline data analysis from multiple files

    Class has a container for data from different files and functions for analysis
    Functions in class can autoload data from specified files according to specified
    paramters
    """

    ######################
    # Define static methods
    #####################

    @staticmethod
    def get_hdf5_path(data_dir):
        """Look for the hdf5 file in the directory

        Args:
            data_dir (str): Path to directory containing HDF5 file

        Raises:
            Exception: If no HDF5 found

        Returns:
            str: Real path to HDF5 file
        """
        hdf5_path = glob.glob(os.path.join(data_dir, "**.h5"))
        if not len(hdf5_path) > 0:
            raise Exception("No HDF5 file detected")
        if len(hdf5_path) > 1:
            selection_list = [
                "{}) {} \n".format(num, os.path.basename(file))
                for num, file in enumerate(hdf5_path)
            ]
            selection_string = "Multiple HDF5 files detected, please select a number:\n{}".format(
                "".join(selection_list)
            )
            file_selection = input(selection_string)
            return hdf5_path[int(file_selection)]

        return hdf5_path[0]

    @staticmethod
    def remove_node(path_to_node, hf5):
        """Remove node from HDF5 if present

        Args:
            path_to_node (str): Path to node
            hf5 (Pytables file obj): Handler for HDF5 file
        """
        if path_to_node in hf5:
            hf5.remove_node(os.path.dirname(path_to_node),
                            os.path.basename(path_to_node))

    ####################
    # Initialize instance
    ###################

    def __init__(self, data_dir=None):
        """Initialize EphysData class

        Args:
            data_dir (str, optional): Directory containing HDF5 file. Defaults to None.
        """
        if data_dir is None:
            self.data_dir = easygui.diropenbox(
                "Please select directory with HDF5 file")
        else:
            self.data_dir = data_dir
            self.hdf5_path = self.get_hdf5_path(data_dir)
            self.hdf5_name = os.path.basename(self.hdf5_path)

            self.spikes = None

    def get_unit_descriptors(self):
        """Extract unit descriptors from HDF5 file"""
        with tables.open_file(self.hdf5_path, "r+") as hf5_file:
            self.unit_descriptors = hf5_file.root.unit_descriptor[:]

    def check_laser(self):
        """Check whether session contains laser variables"""
        with tables.open_file(self.hdf5_path, "r+") as hf5:
            dig_in_list = [x for x in hf5.list_nodes(
                "/spike_trains") if "dig_in" in x.__str__()]

            # Mark whether laser exists or not
            self.laser_durations_exists = (
                sum([dig_in.__contains__("laser_durations")
                    for dig_in in dig_in_list]) > 0
            )

            # If it does, pull out laser durations
            if self.laser_durations_exists:
                self.laser_durations = [dig_in.laser_durations[:]
                                        for dig_in in dig_in_list]

                non_zero_laser_durations = np.sum(self.laser_durations) > 0

            # If laser_durations exists, only non_zero durations
            # will indicate laser
            # If it doesn't exist, then mark laser as absent
            if self.laser_durations_exists:
                if non_zero_laser_durations:
                    self.laser_exists = True
                else:
                    self.laser_exists = False
            else:
                self.laser_exists = False

    def get_spikes(self):
        """Extract spike arrays from specified HD5 files

        Raises:
            Exception: If no spike_trains node found in HDF5 file
        """
        with tables.open_file(self.hdf5_path, "r+") as hf5:
            if "/spike_trains" in hf5:
                dig_in_list = [
                    x for x in hf5.list_nodes("/spike_trains") if "dig_in" in x.__str__()
                ]
            else:
                raise Exception("No spike trains found in HF5")

            self.spikes = [dig_in.spike_array[:] for dig_in in dig_in_list]

    def separate_laser_spikes(self):
        """Separate spike arrays into laser on and off conditions

        Raises:
            Exception: If no laser present for experiment
        """
        if "laser_exists" not in dir(self):
            self.check_laser()
        if "spikes" not in dir(self):
            self.get_spikes()
        if self.laser_exists:
            self.laser_spikes = {}
            self.laser_spikes["on"] = np.array(
                [taste[laser > 0]
                    for taste, laser in zip(self.spikes, self.laser_durations)]
            )
            self.laser_spikes["off"] = np.array(
                [taste[laser == 0]
                    for taste, laser in zip(self.spikes, self.laser_durations)]
            )
        else:
            raise Exception("No laser trials in this experiment")

    def get_region_electrodes(self):
        """If the appropriate json file is present in the data_dir,
        extract the electrodes for each region

        Raises:
            Exception: If .info file cannot be found
        """
        json_path = glob.glob(os.path.join(self.data_dir, "**.info"))[0]
        if os.path.exists(json_path):
            json_dict = json.load(open(json_path, "r"))
            self.region_electrode_dict = json_dict["electrode_layout"]
            self.region_names = [
                x for x in self.region_electrode_dict.keys() if x != "emg"]
        else:
            raise Exception("Cannot find json file. Make sure it's present")

    def get_region_units(self):
        """Extracts indices of units by region of electrodes
        `"""
        if "region_electrode_dict" not in dir(self):
            self.get_region_electrodes()
        if "unit_descriptors" not in dir(self):
            self.get_unit_descriptors()

        unit_electrodes = [x["electrode_number"]
                           for x in self.unit_descriptors]
        region_electrode_vals = [
            val for key, val in self.region_electrode_dict.items() if key != "emg"
        ]

        car_name = []
        car_electrodes = []
        for key, val in self.region_electrode_dict.items():
            if key != "emg":
                for num, this_car in enumerate(val):
                    car_electrodes.append(this_car)
                    car_name.append(key + str(num))

        self.car_names = car_name
        self.car_electrodes = car_electrodes

        car_ind_vec = np.zeros(len(unit_electrodes))
        for num, val in enumerate(self.car_electrodes):
            for elec_num, elec in enumerate(unit_electrodes):
                if elec in val:
                    # This tells you which car group each neuron is in
                    car_ind_vec[elec_num] = num

        self.car_units = [np.where(car_ind_vec == x)[0]
                          for x in np.unique(car_ind_vec)]

        region_ind_vec = np.zeros(len(unit_electrodes))
        for elec_num, elec in enumerate(unit_electrodes):
            for region_num, region in enumerate(region_electrode_vals):
                for car in region:
                    if elec in car:
                        region_ind_vec[elec_num] = region_num

        self.region_units = [np.where(region_ind_vec == x)[0]
                             for x in np.unique(region_ind_vec)]

    def return_region_spikes(self, region_name="all", laser=None):
        """Use metadata to return spike trains by region

        Args:
            region_name (str, optional): Extract spike train from specified region.
            Defaults to 'all'.

        Raises:
            Exception: If region name not found in .info file or more than one match found

        Returns:
            Numpy array: Array containing spike trains for specified region
        """
        if laser not in [None, "on", "off"]:
            raise Exception('laser must be from ["on","off"]')
        if "region_names" not in dir(self):
            self.get_region_units()
        if self.spikes is None:
            self.get_spikes()

        if laser is not None:
            self.separate_laser_spikes()
            this_spikes = self.laser_spikes[laser]
        else:
            this_spikes = self.spikes

        if region_name != "all":
            region_ind = [num for num, x in enumerate(
                self.region_names) if x == region_name]
            if not len(region_ind) == 1:
                raise Exception(
                    "Region name not found, or too many matches found, "
                    "acceptable options are" + "\n" +
                    f"===> {self.region_names, 'all'}"
                )

            this_region_units = self.region_units[region_ind[0]]
            # region_spikes = [x[:, this_region_units] for x in self.spikes]
            region_spikes = [x[:, this_region_units] for x in this_spikes]
            return np.array(region_spikes)

        return np.array(this_spikes)
        # return np.array(self.spikes)
