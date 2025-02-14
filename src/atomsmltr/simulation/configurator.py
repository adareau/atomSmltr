# -*- coding: utf-8 -*-
"""Defines the Configuration classes,
to make a consistent configuration for the simulator
"""

# % IMPORTS
from copy import copy
import warnings

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
            "zone": self.__zones,
        }

        # - init atom
        if atom is not None:
            self.atom = atom

        if object_list is not None:
            self.add_objects(object_list)

    # -- COLLECTION HANDLING METHOD

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
    def print_object_info(self, collection, tag):
        """Print info for an object from 'collection' with 'tag'

        Collection must be in ['laser', 'magnetic field', 'zone'    ]

        Args:
            collection (str): the collection from which the object should be removed
            tag (str): the tag of the object
        """
        coll = self.__check_object_in_coll(collection, tag)
        info = coll[tag].gen_infostring_obj()
        info.title = f"{collection} | {tag=}"
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
        if not isinstance(atom, Atom):
            raise TypeError("'atom' should be an atom")
        self.__atom = atom
        # TODO here, check sanity of __atomlight ; warning
