"""configuration
==================

Here we implement the ``Configuration`` class, that allows to define
consistent configurations that are later fed to the ``Simulation`` objects.

Quick description
--------------------

A configuration consists of:

* a atom (``Atom``)
* a collection of laser beams (``LaserBeam``)
* a collection of magnetic Fields (``MagneticField``)
* a collection of forces (``Force``)
* a collection of zones (``Zone``)

The coupling between atoms and lasers is stored in a ``atomlight`` dictionnary, that
is setup with the ``add_atomlight_coupling()`` method.
"""

# % IMPORTS
import warnings
import numpy as np
from copy import copy, deepcopy

# % LOCAL IMPORTS
from ..environment import LaserBeam, MagneticField, Zone, SuperZone, Force
from ..environment.envbase import EnvObject
from ..atoms import Atom
from ..utils.infostring import InfoString

# % CONSTANTS

SEP_STR = "# ------------ {} ------------ #"

# % DEFINE THE CLASS


class Configuration(object):
    """Defines a configuration for the simulation

    Parameters
    ----------
    object_list : EnvObject | list, optional
        list of environment objects (lasers, magnetic fields, zones, forces)
        to include in the configuration, by default None
    atom : Atom, optional
        atom for the simulation, by default None

    Examples
    ---------

    .. code-block:: python

        # - imports
        from atomsmltr.environment import GaussianLaserBeam, MagneticOffset, Limits
        from atomsmltr.atoms import Ytterbium
        from atomsmltr.simulation import Configuration

        # - setup environment objects
        laser = GaussianLaserBeam(tag="laser")
        mag_offset = MagneticOffset(offset=(0,1,0), tag="offset")
        xlim = Limits(min=0, max=10, axis=0, target="position")

        # - setup atom
        yb = Ytterbium()

        # - init configuration
        config = Configuration(object_list=[laser, mag_offset, xlim], atom=yb)

        # - print info
        config.print_info()


    """

    def __init__(self, object_list: EnvObject | list = None, atom: Atom = None):

        # - initialize collections
        self.__lasers = {}
        self.__zones = {}
        self.__magfields = {}
        self.__forces = {}
        self.__atomlight = {}
        self.__atom = None

        self.__implemented_collections = {
            "laser": self.__lasers,
            "magnetic field": self.__magfields,
            "force": self.__forces,
            "zone": self.__zones,
        }

        # - init atom
        if atom is not None:
            self.atom = atom

        if object_list is not None:
            self.add_objects(object_list)

    # -- ATOM-LIGHT INTERACTION HANDLING

    def get_atomlight_couples(self) -> list:
        """Returns a list of (transition, laser, detuning) tuples

        Returns
        -------
        list
            a list of tuples with (transition, laser, detuning)
        """
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
        laser: str | LaserBeam,
        transition: str,
        detuning: float,
        verbose: bool = False,
        override: bool = False,
    ):
        """Adds a atom-light coupling element in the configuration

        Parameters
        ----------
        laser : str | LaserBeam
            either a laser tag or a laser object. This object/tag has to be in the configuration laser list
        transition : str
            the tag of the transition. Should be part of the collection's atom transition list
        detuning : float
            detuning of the laser w.r.t to transition (rad/s), see notes for the definition
        verbose : bool, optional
            if True, will print messages when adding the couplint, by default False
        override : bool, optional
            if set to True, if a coupling between the laser and the transition already
            exists, then it will be overriden. Otherwise it will raise an error, by default False

        Notes
        ------

        The detuning δ is defined as:

            δ = ωL - ω0

        Where ωL is the laser pulsation and ω0 the atomic transition pulsation (hence, in rad/s)
        Stated otherwise, detuning units is in units of 2π x Hz
        """
        # - checking inputs
        # check laser argument
        if not isinstance(laser, (str, LaserBeam)):
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
        laser: str | LaserBeam,
        transition: str,
    ):
        """Removes an atom-light coupling

        Parameters
        ----------
        laser : str | LaserBeam
            laser coupled : tag (str) or directly the object
        transition : str
            transition tag
        """
        # - checking inputs
        # check laser argument
        if not isinstance(laser, (str, LaserBeam)):
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
    def getB(self, position: np.ndarray) -> np.ndarray:
        """Returns the total magnetic field at a given position in the lab frame

        Parameters
        ----------
        position : array, shape (3,) or (n1, n2, .., 3)
            array of cartesian coordinates in the lab frame

        Returns
        -------
        B : array, shape (3,) or (n1, n2, .., 3)
            magnetic field at the position. shape matches the one of ``position``

        Notes
        ------
        position is an array_like object, with shape (3,) or (n1, n2, .., 3).

        In all cases, the last dimension contains cordinates (x, y, z),
        in meter and in the lab frame

        Example
        -------

        .. code-block:: python

            ... init a proper config first
            import numpy as np

            # generate a grid of 100 x 100 points in the (x, y) plane
            grid = np.mgrid[-10:10:100j, -10:-10:100j, 0:0:1j]
            grid = np.squeeze(grid)

            # get coordinates arrays (for plotting for instance)
            X, Y, Z = grid

            # generate the requested (..., 3) shaped position array
            position = grid.T

            # compute magnetic field
            B = config.getB(position)

            # get magnetic field components
            Bx, By, Bz = B.T

            # show shapes (to illustrate what we did)
            print(f"{grid.shape=}")
            print(f"{position.shape=}")
            print(f"{B.shape=}")
            print(f"{X.shape=}")
            print(f"{Bx.shape=}")

        This returns

        .. code-block:: python

            grid.shape=(3, 100, 100)
            position.shape=(100, 100, 3)
            B.shape=(100, 100, 3)
            X.shape=(100, 100)
            Bx.shape=(100, 100)

        """
        B = np.zeros_like(position, dtype=float)
        if self.__magfields:
            for magfield in self.__magfields.values():
                B += magfield.get_value(position)
        return B

    def getBnorm(self, position):
        """Returns the magnetic field amplitude (norm) at a given lab position

        Parameters
        ----------
        position : array, shape (3,) or (n1, n2, .., 3)
            array of cartesian coordinates in the lab frame

        Returns
        -------
        B_norm : array, shape (1,) or (n1, n2, .., 1)
            magnetic field norm the position. shape matches the one of ``position``

        Notes
        ------
        position is an array_like object, with shape (3,) or (n1, n2, .., 3).

        In all cases, the last dimension contains cordinates (x, y, z),
        in meter and in the lab frame
        """
        B = self.getB(position)
        Bx, By, Bz = B.T
        B_norm = np.sqrt(Bx**2 + By**2 + Bz**2).T
        return B_norm

    # -- GETTING ZONES
    def get_stop_zones(self):
        """Returns two list of the zones whose ``action`` are set to ``stop``

        Returns
        -------
        stop_position: list
            list of position stop zones (target=position)
        stop_speed: list
            list of speed stop zones (target=speed)
        """

        return self._get_zones(action="stop")

    def get_all_zones(self):
        """Returns zones sorted in according to their target

        Returns
        -------
        stop_position: list
            list of position stop zones (target=position)
        stop_speed: list
            list of speed stop zones (target=speed)
        """

        return self._get_zones(action="all")

    def _get_zones(self, action: str = "stop"):
        """Returns two list of the zones whose ``action`` are set to a given value

        Parameters
        ----------
        action : str, optional
            action to target, by default "stop"

        Returns
        -------
        stop_position: list
            list of position zones (target=position)
        stop_speed: list
            list of speed zones (target=speed)
        """
        stop_speed = []
        stop_position = []
        for zone in self.__zones.values():
            if zone.action == action or action == "all":
                if zone.target == "speed":
                    stop_speed.append(deepcopy(zone))
                elif zone.target == "position":
                    stop_position.append(deepcopy(zone))
        return stop_position, stop_speed

    def in_zone(self, pos_speed_vector: np.ndarray, action: str = "stop") -> np.ndarray:
        """Evaluates whether 'pos_speed_vector' is in the zones corresponding to a given action

        Parameters
        ----------
        vector : array of shape (6,) or (n1, n2, ..., 6)
            cartesian coordinates of the vectors in the lab frame
        action : str, optionnal
            the action of the zones to consider by default "stop"

        Returns
        -------
        in_zone : array of shape (1,) or (n1, n2, ..., 1)
            whether the vector is 'in the zone'

        Notes
        -----

        ``vector`` should be an array of shape (6,) or (n1, n2, .., 6), where last axis contains
        the coordinates (position & speed) to evaluate.

        In all cases, the last dimension contains cordinates (x, y, z, vx, vy, vz),
        """

        # -- init a SuperZone
        collection = SuperZone(logic="AND")

        # -- populate
        for zone in self.__zones.values():
            if zone.action == action:
                collection += zone

        # -- return results
        return collection.get_value(pos_speed_vector)

    # -- GETTING FORCES

    def get_all_forces(self) -> list:
        """Returns a list of all forces

        Returns
        -------
        force_list (list)
            a list of all forces in the configuration
        """
        force_list = self.__forces.values()
        return force_list

    # -- COLLECTION HANDLING METHODS

    # ADDING
    def add_objects(self, obj: EnvObject | list, verbose=False):
        """Add environment objects to the configuration.

        Parameters
        ----------
        obj : EnvObject | list
            a environment object or a list of objects
        verbose : bool, optional
            if set to True messages are displayed. Defaults to False.

        Notes
        -----

        The function takes a single environment object (laser, magnetic field...) or a collection
        of objects in the form of a tuple or a list.

        Objects of different subtypes can be added at the same time: the method will add them
        to the correct collection based on their classes

        Note
        ----
            The addition operator ``+`` also allows to add objects.
            Hence, ``conf.add_objects([obj1, obj2, ...])`` is equivalent to
            ``conf += obj1, obj2 , ...``

        Examples
        --------
        .. code-block:: python

            ... init a proper config first and env objects

            # add objects
            config.add_objects(laser1)
            config.add_objects([mag_field, zone1, zone2, laser2])

            # also works with += operator*
            config += laser3, laser4

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
        elif isinstance(obj, LaserBeam):
            collection = self.__lasers
            name = "lasers"
        elif isinstance(obj, Force):
            collection = self.__forces
            name = "forces"
        elif isinstance(obj, Zone):
            collection = self.__zones
            name = "zones"
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

        Parameters
        ----------
        obj : EnvObject | list
            a environment object or a list of objects
        verbose : bool, optional
            if set to True messages are displayed. Defaults to False.
        error_on_fail : bool, optional
            if set to True, raises an error if it fails. Otherwise,
            just raises a warning and continues. Defaults to False.

        Notes
        ------
        The function takes a single environment object (laser, magnetic field...) or a collection
        of objects in the form of a tuple or a list.

        For each object given as an input, if there is an object with:

        (1) same type (laser, magnetic field) **and**
        (2) same tag

        then this object is replaced by the new one.
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
        elif isinstance(obj, Force):
            collection = self.__forces
            name = "forces"
        elif isinstance(obj, Zone):
            collection = self.__zones
            name = "zones"
        elif isinstance(obj, LaserBeam):
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
        """Returns the list of zones' tags in the current config"""
        return list(self.__zones)

    def list_forces(self):
        """Returns the list of forces' tags in the current config"""
        return list(self.__forces)

    # REMOVING
    def rm_object(self, collection: str, tag: str):
        """Remove object from 'collection' with 'tag'

        Collection must be in ['laser', 'magnetic field', 'zone', 'force']

        Parameters
        ----------
        collection : str
            the collection from which the object should be removed
        tag : str
            the tag of the object
        """

        coll = self.__check_object_in_coll(collection, tag)
        del coll[tag]

    def rm_laser(self, tag: str):
        """Removes laser by tag

        Parameters
        ----------
        tag : str
            laser tag
        """
        return self.rm_object("laser", tag)

    def rm_magnetic_field(self, tag):
        """Removes magnetic field by tag

        Parameters
        ----------
        tag : str
            magnetic field tag
        """
        return self.rm_object("magnetic field", tag)

    def rm_zone(self, tag):
        """Removes zone by tag

        Parameters
        ----------
        tag : str
            zone tag
        """
        return self.rm_object("zone", tag)

    def rm_force(self, tag):
        """Removes force by tag

        Parameters
        ----------
        tag : str
            force tag
        """
        return self.rm_object("force", tag)

    def rm_all_objects(self):
        """Remove all objects"""
        self.rm_all_lasers()
        self.rm_all_magnetic_fields()
        self.rm_all_zones()
        self.rm_all_forces()

    def rm_all_lasers(self):
        """Remove all lasers"""
        self.__lasers.clear()

    def rm_all_magnetic_fields(self):
        """Remove all magnetic fields"""
        self.__magfields.clear()

    def rm_all_zones(self):
        """Remove all zones"""
        self.__zones.clear()

    def rm_all_forces(self):
        """Remove all forces"""
        self.__forces.clear()

    # -- INFOS

    def gen_object_infostring_object(self, collection: str, tag: str) -> InfoString:
        """Generate infostring object for an object from 'collection' with 'tag'

        Collection must be in ['laser', 'magnetic field', 'zone']

        Parameters
        ----------
        collection : str
            the collection from which the object should be taken
        tag : str
            the tag of the object

        Returns
        -------
        infostring: Infostring
            an infostring object

        See also
        ---------
        atomsmltr.utils.infostring
        """

        coll = self.__check_object_in_coll(collection, tag)
        info = coll[tag].gen_infostring_obj()
        info.title = f"{collection} | {tag=}"
        return info

    def print_object_info(self, collection: str, tag: str):
        """Print info for an object from 'collection' with 'tag'

        Collection must be in ['laser', 'magnetic field', 'zone'    ]

        Parameters
        ----------
        collection : str
            the collection from which the object should be taken
        tag : str
            the tag of the object
        """
        info = self.gen_object_infostring_object(collection, tag)
        print(info.generate())

    def print_laser_info(self, tag: str):
        """Print info of the laser indentified by 'tag'

        Parameters
        ----------
        tag : str
            the tag of the laser
        """
        return self.print_object_info("laser", tag)

    def print_magnetic_field_info(self, tag: str):
        """Print info of the magnetic field indentified by 'tag'

        Parameters
        ----------
        tag : str
            the tag of the magnetic field
        """
        return self.print_object_info("magnetic field", tag)

    def print_zone_info(self, tag: str):
        """Print info of the zone indentified by 'tag'

        Parameters
        ----------
        tag : str
            the tag of the zone
        """
        return self.print_object_info("zone", tag)

    def print_force_info(self, tag: str):
        """Print info of the force indentified by 'tag'

        Parameters
        ----------
        tag : str
            the tag of the force
        """
        return self.print_object_info("force", tag)

    # -- GET OBJECTS
    def get_object_copy(self, collection: str, tag: str) -> EnvObject:
        """Returns a copy of an object from 'collection' with 'tag'

        Collection must be in ['laser', 'magnetic field', 'zone', 'force' ]

        Parameters
        ----------
        collection : str
            the collection from which the object should be taken
        tag : str
            the tag of the object
        """
        coll = self.__check_object_in_coll(collection, tag)
        return copy(coll[tag])

    def get_laser_copy(self, tag: str):
        """Returns a copy of the laser indentified by 'tag'

        Parameters
        ----------
        tag : str
            the tag of the laser
        """
        return self.get_object_copy("laser", tag)

    def get_magnetic_field_copy(self, tag: str):
        """Returns a copy of the magnetic field indentified by 'tag'

        Parameters
        ----------
        tag : str
            the tag of the magnetic field
        """
        return self.get_object_copy("magnetic field", tag)

    def get_zone_copy(self, tag: str):
        """Returns a copy of the zone indentified by 'tag'

        Parameters
        ----------
        tag : str
            the tag of the zone
        """
        return self.get_object_copy("zone", tag)

    def get_force_copy(self, tag: str):
        """Returns a copy of the force indentified by 'tag'

        Parameters
        ----------
        tag : str
            the tag of the force
        """
        return self.get_object_copy("force", tag)

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
        """ "Atom: the configuration atom"""
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

    @property
    def objects(self) -> dict:
        """dict: the collection of objects"""
        out = {}
        for k, v in self.__implemented_collections.items():
            out[k] = copy(v)
        return out

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
        """Prints atom-light coupling information"""
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

    # -- OPERATORS OVERLOADING

    def __add__(self, object: EnvObject):
        """returns a copy of the configuration, with the object added"""
        new_config = deepcopy(self)
        new_config.add_objects(object)
        return new_config

    def __iadd__(self, object: EnvObject):
        """adds the objects"""
        self.add_objects(object)
        return self

    def __imod__(self, object: EnvObject):
        """updates the objects"""
        self.update_objects(object)
        return self
