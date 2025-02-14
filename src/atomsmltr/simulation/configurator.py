# -*- coding: utf-8 -*-
"""Defines the Configuration classes,
to make a consistent configuration for the simulator
"""

# % IMPORTS
from copy import copy

# % LOCAL IMPORTS
from ..environment import AbstractLaserBeam, MagneticField
from ..environment.envbase import EnvObject
from ..atoms import Atom


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
            "zones": self.__zones,
        }

        # - init atom
        if atom is not None:
            self.atom = atom

        if object_list is not None:
            self.add_objects(object_list)

    # -- COLLECTION HANDLING METHOD

    # ADDING
    def add_objects(self, obj: EnvObject | list, verbose=False):
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
        elif isinstance(obj, AbstractLaserBeam):
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
        # - check that object tag not present
        msg = f"We already have an element with tag '{obj.tag}' in our {name} collection. "
        msg += "Remove or update this element."
        if obj.tag in collection:
            raise ValueError(msg)
        # - add the object
        collection[obj.tag] = obj

    # UPDATING
    def update_objects(self, obj: EnvObject | list, verbose=False, error_on_fail=False):
        # - check argument
        self.__check_objects_arg(obj)

        # - recursive add if list
        if isinstance(obj, (list, tuple)):
            for element in obj:
                self.update_objects(element, verbose)
            return

        # - add object
        if isinstance(obj, MagneticField):
            collection = self.__magfields
            name = "magnetic fields"
        elif isinstance(obj, AbstractLaserBeam):
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
        # - check that object tag not present
        msg = f"There is no element with tag '{obj.tag}' in our {name} collection. "
        if not obj.tag in collection:
            raiser = ValueError if error_on_fail else Warning
            raiser(msg)
            return False
        # - update the object
        collection[obj.tag] = obj
        return True

    # LISTING
    def list_lasers(self):
        return list(self.__lasers)

    def list_magnetic_fields(self):
        return list(self.__magfields)

    # REMOVING
    def rm_object(self, collection, tag):
        coll = self.__check_object_in_coll(collection, tag)
        del coll[tag]

    def rm_laser(self, tag):
        return self.rm_object("laser", tag)

    def rm_magnetic_field(self, tag):
        return self.rm_object("magnetic field", tag)

    def rm_all_objects(self):
        self.__lasers = {}
        self.__magfields = {}
        self.__zones = {}

    def rm_all_lasers(self):
        self.__lasers = {}

    def rm_all_magnetic_fields(self):
        self.__magfields = {}

    def rm_all_zones(self):
        self.__zones = {}

    # -- INFOS
    def print_object_info(self, collection, tag):
        coll = self.__check_object_in_coll(collection, tag)
        coll[tag].print_info()

    def print_laser_info(self, tag):
        return self.print_object_info("laser", tag)

    def print_magnetic_field_info(self, tag):
        return self.print_object_info("magnetic field", tag)

    # -- GET OBJECTS
    def get_object_copy(self, collection, tag) -> EnvObject:
        coll = self.__check_object_in_coll(collection, tag)
        return copy(coll[tag])

    def get_laser_copy(self, tag):
        return self.get_object_copy("laser", tag)

    def get_magnetic_field_copy(self, tag):
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
        if not isinstance(atom, Atom):
            raise TypeError("'atom' should be an atom")
        self.__atom = atom
        # TODO here, check sanity of __atomlight ; warning
