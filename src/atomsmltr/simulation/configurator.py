# -*- coding: utf-8 -*-
"""Defines the Configuration classes,
to make a consistent configuration for the simulator
"""

# % IMPORTS
import warnings
import numpy as np
from copy import copy

# % LOCAL IMPORTS
from ..environment import BaseLaserBeam, MagneticField
from ..environment.envbase import EnvObject
from ..atoms import Atom
from ..utils.infostring import InfoString

# % CONSTANTS

SEP_STR = "# ------------ {} ------------ #"

# % DEFINE THE CLASS


class Configuration(object):
    def __init__(self, object_list=None, atom=None):
        # - initialize collections
        self.__lasers = {}
        self.__zones = {}
        self.__magfields = {}
        self.__atomlight = {}
        self.__atom = None

        self.__implemented_collections = {
            "laser": self.__lasers,
            "magnetic field": self.__magfields,
            "zone": self.__zones,
        }

        # - init atom
        if atom is not None:
            self.atom = atom

        if object_list is not None:
            self.add_objects(object_list)

    # -- ATOM-LIGHT INTERACTION HANDLING

    def get_atomlight_couples(self):
        list = []
        for transition_tag, laser_dict in self.__atomlight.items():
            transition = self.atom.trans[transition_tag]
            for laser_tag, coupling_info in laser_dict.items():
                laser = self.__lasers[laser_tag]
                detuning = coupling_info["detuning"]
                list.append((transition, laser, detuning))
        return list

    def add_atomlight_coupling(
        self,
        laser: str | BaseLaserBeam,
        transition: str,
        detuning: float,
        verbose=False,
        override=False,
    ):
        # - checking inputs
        # check laser argument
        if not isinstance(laser, (str, BaseLaserBeam)):
            raise TypeError("'laser' should be a tag (string) or a Laser object")
        if not isinstance(laser, str):
            laser = laser.tag
        # check that laser is there
        if laser not in self.__lasers:
            msg = f"No entry for laser tag '{laser}'. "
            msg += f" Available lasers are {list(self.__lasers)}."
            raise KeyError(msg)
        # check that transition is there
        if self.atom is None:
            raise ValueError("No atom was defined for this config")
        if transition not in self.__atomlight:
            msg = f"No entry for transition '{transition}'. "
            msg += f" Available transitions are {list(self.__atomlight)}."
            raise KeyError(msg)

        # - check that there is no link
        if laser in self.__atomlight[transition]:
            msg = f"There is alreay a link between laser '{laser}' and transition '{transition}'. "
            if not override:
                msg += "Since 'override' is set to 'False', we stop here with an error."
                raise KeyError(msg)
            else:
                msg += "Since 'override' is set to 'True', we go on."
                if verbose:
                    print(" > " + msg)
        # - store
        self.__atomlight[transition][laser] = {"detuning": detuning}

    def rm_atomlight_coupling(
        self,
        laser: str | BaseLaserBeam,
        transition: str,
    ):
        # - checking inputs
        # check laser argument
        if not isinstance(laser, (str, BaseLaserBeam)):
            raise TypeError("'laser' should be a tag (string) or a Laser object")
        if not isinstance(laser, str):
            laser = laser.tag

        # - remove
        success = False
        if transition in self.__atomlight:
            if laser in self.__atomlight[transition]:
                self.__atomlight[transition].pop(laser)
                success = True
        if not success:
            msg = f"There is no link between '{laser}' and '{transition}'."
            raise KeyError(msg)

    def reset_atomlight_coupling(self):
        for transition in self.__atomlight:
            self.__atomlight[transition].clear()

    # -- GETTING VALUES
    def getB(self, position):
        """Returns magnetic field at a given position in the lab frame

            position is an array_like object, with shape (3,) or (n1, n2, .., 3).
            In all cases, the last dimension contains cordinates (x, y, z), in meter and in the lab frame

        Args:
            position (array_like, shape (3,) or (n,3)) : positions at which the intensity is computed

        Returns:
            magnetic field (float or array): laser intensity at positions, with dimension matching the 'position' input.
        """
        B = np.zeros_like(position, dtype=float)
        if self.__magfields:
            for magfield in self.__magfields.values():
                B += magfield.value(position)
        return B

    def getBnorm(self, position):
        """Returns magnetic field norm at a given position in the lab frame

            position is an array_like object, with shape (3,) or (n1, n2, .., 3).
            In all cases, the last dimension contains cordinates (x, y, z), in meter and in the lab frame

        Args:
            position (array_like, shape (3,) or (n,3)) : positions at which the intensity is computed

        Returns:
            magnetic field (float or array): laser intensity at positions, with dimension matching the 'position' input.
        """
        B = self.getB(position)
        Bx, By, Bz = B.T
        B_norm = np.sqrt(Bx**2 + By**2 + Bz**2).T
        return B_norm

    # -- COLLECTION HANDLING METHODS

    # ADDING
    def add_objects(self, obj: EnvObject | list, verbose=False):
        """Add environment objects to the configuration.

        The function takes a single environment object (laser, magnetic field...) or a collection
        of objects in the form of a tuple or a list.

        Objects of different subtypes can be added at the same time: the method will add them
        to the correct collection based on their classes

        Args:
            obj (EnvObject | list): a environment object or a list of objects
            verbose (bool, optional): if set to True messages are displayed. Defaults to False.
        """
        # - check argument
        self.__check_objects_arg(obj)

        # - recursive add if list
        if isinstance(obj, (list, tuple)):
            for element in obj:
                self.add_objects(element, verbose)
            return

        # - add object
        if isinstance(obj, MagneticField):
            collection = self.__magfields
            name = "magnetic fields"
        elif isinstance(obj, BaseLaserBeam):
            collection = self.__lasers
            name = "lasers"
        else:
            msg = f"Objects of type {type(obj)} are not handled yet.. where did you find this ?"
            raise TypeError(msg)
        self.__add_obj(obj, collection, name)

        if verbose:
            msg = f"(+) sucessfully added object '{obj.tag}' in the {name} collection"
            print(msg)

    def __add_obj(self, obj, collection, name):
        """Internal method to add objects"""
        # - copy
        obj = copy(obj)
        # - check that object tag not present
        msg = f"We already have an element with tag '{obj.tag}' in our {name} collection. "
        msg += "Remove or update this element."
        if obj.tag in collection:
            raise ValueError(msg)
        # - add the object >>> we use a copy to avoid unwanted modifications
        collection[obj.tag] = obj

    # UPDATING
    def update_objects(self, obj: EnvObject | list, verbose=False, error_on_fail=False):
        """Update an object or a list of objects

        The function takes a single environment object (laser, magnetic field...) or a collection
        of objects in the form of a tuple or a list.

        For each object given as an input, if there is an object with:
            (1) same type (laser, magnetic field)
        AND (2) same tag

        then this object is replaced by the new one.

        Args:
            obj (EnvObject | list): a environment object or a list of objects
            verbose (bool, optional): if set to True messages are displayed. Defaults to False.
            error_on_fail (bool, optional): If set to True, raises an error if it fails. Otherwise, just raises a warning and continues. Defaults to False.

        Raises:
            TypeError: _description_
        """
        # - check argument
        self.__check_objects_arg(obj)

        # - recursive add if list
        if isinstance(obj, (list, tuple)):
            for element in obj:
                self.update_objects(element, verbose, error_on_fail)
            return
        # - add object
        if isinstance(obj, MagneticField):
            collection = self.__magfields
            name = "magnetic fields"
        elif isinstance(obj, BaseLaserBeam):
            collection = self.__lasers
            name = "lasers"
        else:
            msg = f"Objects of type {type(obj)} are not handled yet.. where did you find this ?"
            raise TypeError(msg)
        success = self.__upd_obj(obj, collection, name, error_on_fail)
        if verbose:
            if success:
                msg = f"(>) sucessfully updated object '{obj.tag}' in the {name} collection"
            else:
                msg = f"(x) could not update '{obj.tag}' in the {name} collection"
            print(msg)

    def __upd_obj(self, obj, collection, name, error_on_fail) -> bool:
        # - copy
        obj = copy(obj)
        # - check that object tag not present
        msg = f"There is no element with tag '{obj.tag}' in our {name} collection. "
        if not obj.tag in collection:
            if error_on_fail:
                raise KeyError(msg)
            else:
                warnings.warn(msg)
            return False
        # - update the object
        collection[obj.tag] = obj
        return True

    # LISTING
    def list_lasers(self) -> list:
        """Returns the list of laser's tags in the current config"""
        return list(self.__lasers)

    def list_magnetic_fields(self):
        """Returns the list of magnetic fields' tags in the current config"""
        return list(self.__magfields)

    def list_zones(self):
        """Returns the list of magnetic fields' tags in the current config"""
        return list(self.__zones)

    # REMOVING
    def rm_object(self, collection: str, tag: str):
        """Remove object from 'collection' with 'tag'

        Collection must be in ['laser', 'magnetic field', 'zone'    ]

        Args:
            collection (str): the collection from which the object should be removed
            tag (str): the tag of the object
        """
        coll = self.__check_object_in_coll(collection, tag)
        del coll[tag]

    def rm_laser(self, tag: str):
        """Remove lasers by tag

        Args:
            tag (str): laser tag
        """
        return self.rm_object("laser", tag)

    def rm_magnetic_field(self, tag):
        """Remove magnetic fields by tag

        Args:
            tag (str): magnetic field tag
        """
        return self.rm_object("magnetic field", tag)

    def rm_all_objects(self):
        """Remove all objects"""
        self.rm_all_lasers()
        self.rm_all_magnetic_fields()
        self.rm_all_zones()

    def rm_all_lasers(self):
        """Remove all lasers"""
        self.__lasers.clear()

    def rm_all_magnetic_fields(self):
        """Remove all magnetic fields"""
        self.__magfields.clear()

    def rm_all_zones(self):
        """Remove all zones"""
        self.__zones.clear()

    # -- INFOS

    def gen_object_infostring_object(self, collection, tag):
        """Generate infostring object for an object from 'collection' with 'tag'

        Collection must be in ['laser', 'magnetic field', 'zone'    ]

        Args:
            collection (str): the collection from which the object should be removed
            tag (str): the tag of the object
        """
        coll = self.__check_object_in_coll(collection, tag)
        info = coll[tag].gen_infostring_obj()
        info.title = f"{collection} | {tag=}"
        return info

    def print_object_info(self, collection, tag):
        """Print info for an object from 'collection' with 'tag'

        Collection must be in ['laser', 'magnetic field', 'zone'    ]

        Args:
            collection (str): the collection from which the object should be removed
            tag (str): the tag of the object
        """
        info = self.gen_object_infostring_object(collection, tag)
        print(info.generate())

    def print_laser_info(self, tag):
        """Print info of the laser indentified by 'tag'

        Args:
            tag (str): laser tag
        """
        return self.print_object_info("laser", tag)

    def print_magnetic_field_info(self, tag):
        """Print info of the magnetic field indentified by 'tag'

        Args:
            tag (str): magnetic field tag
        """
        return self.print_object_info("magnetic field", tag)

    # -- GET OBJECTS
    def get_object_copy(self, collection, tag) -> EnvObject:
        """Returns a copy of an object from 'collection' with 'tag'

        Collection must be in ['laser', 'magnetic field', 'zone'    ]

        Args:
            collection (str): the collection from which the object should be removed
            tag (str): the tag of the object
        """
        coll = self.__check_object_in_coll(collection, tag)
        return copy(coll[tag])

    def get_laser_copy(self, tag):
        """Returns a copy of the laser indentified by 'tag'

        Args:
            tag (str): laser tag
        """
        return self.get_object_copy("laser", tag)

    def get_magnetic_field_copy(self, tag):
        """Returns a copy of the magnetic field indentified by 'tag'

        Args:
            tag (str): magnetic field tag
        """
        return self.get_object_copy("magnetic field", tag)

    # -- COMMON METHODS

    def __check_object_in_coll(self, collection, tag) -> dict:
        implemented_collections = self.__implemented_collections
        if collection not in implemented_collections:
            msg = f"Wrong collection. implemented collections are {list(implemented_collections)}"
            raise ValueError(msg)

        coll = implemented_collections[collection]
        if tag not in coll:
            msg = f"There is no {collection} with tag {tag}"
            raise KeyError(msg)
        return coll

    def __check_objects_arg(self, obj):
        """Called in add_objects & update_objects"""
        type_err_msg = "passed argument should be an EnvObject or a list of EnvObjects"
        if isinstance(obj, (list, tuple)):
            for element in obj:
                if not isinstance(element, EnvObject):
                    raise TypeError(type_err_msg)
        elif not isinstance(obj, EnvObject):
            raise TypeError(type_err_msg)

    # -- GETTERS & SETTERS
    @property
    def atom(self) -> Atom:
        return self.__atom

    @atom.setter
    def atom(self, atom: Atom):
        # - set atom
        if not isinstance(atom, Atom):
            raise TypeError("'atom' should be an atom")
        self.__atom = atom
        # - prepare atomlight dict
        # issue warning if already some entries
        if not self.__atomlight:
            # if dict not empty, clear it
            self.__atomlight.clear()
            # warnings.warn("Resetting atom-light dictionnary...")
        for transition_tag in self.atom.list_transitions():
            self.__atomlight[transition_tag] = {}

    # -- INFO PRINTER
    def gen_atomlight_infostring_obj(self):
        info = InfoString("Atom-light couplings")
        for transition, couplings in self.__atomlight.items():
            info.add_section(f"transition > '{transition}'")
            if couplings:
                for laser, params in couplings.items():
                    detuning = params["detuning"]
                    trans_Gamma = self.atom.trans[transition].Gamma
                    det_str = f"{detuning=:.3g}"
                    det_str += f" ({detuning / trans_Gamma:.2f}Γ)"
                    info.add_element(f"laser '{laser}'", det_str)
            else:
                info.add_element("empty")
        return info

    def print_atomlight_info(self):
        print(self.gen_atomlight_infostring_obj().generate())

    def gen_infostring_obj_list(self):
        # - prepare output
        info_list = []
        # - general infostring
        info = InfoString("General informations")
        # atom
        info.add_section("atom")
        info.add_element("name", self.atom.name)
        # collections
        for name, coll in self.__implemented_collections.items():
            info.add_section(name + "s")
            if coll:
                for tag in coll:
                    info.add_element(tag)
            else:
                info.add_element("empty")

        # append to list
        info_list.append(info)

        # - atom info
        info = self.atom.gen_infostring_obj()
        info.title = f"atom | {self.atom.name.lower()}"
        info_list.append(info)

        # - collections
        for name, coll in self.__implemented_collections.items():
            for tag in coll:
                info = self.gen_object_infostring_object(name, tag)
                info_list.append(info)

        # - atom light
        info = self.gen_atomlight_infostring_obj()
        info_list.append(info)

        return info_list

    def print_info(self):
        """Prints informations on the configuration"""
        info_list = self.gen_infostring_obj_list()
        print(SEP_STR.format("CONFIG INFO > START"))
        for info in info_list:
            print(info.generate())
        print(SEP_STR.format("CONFIG INFO > STOP "))
