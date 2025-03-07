"""infostrings
==================

Here we implement the ``InfoString`` class, that allows to gather
different parameters from objects and to display them in a tree format.

This is used in the ``print_info()`` method that is present in most
classes accross the module.

Examples
--------------------

Run a simulation with one initial condition vector

.. code-block:: python

        from atomsmltr.utils.infostring import InfoString

        info = InfoString("PARAMETERS")
        info.add_section("First Section")
        info.add_element("elem", "param")
        info.add_element("x", f"{25.4e-6:.2e}µm")
        info.add_element("y", f"{15.8:.2f} mW")
        info.add_section("Another section")
        info.add_element("elem", "param")
        info.add_element("elem2", "param2")

        print(info.generate())

This prints :

.. code-block::

        ──────────────
        | PARAMETERS |
        ──────────────
        . First Section :
        ├── elem : param
        ├── x : 2.54e-05µm
        └── y : 15.80 mW

        . Another section :
        ├── elem : param
        └── elem2 : param2

"""

# % IMPORTS

from collections import OrderedDict

# % CONSTANTS

HEADER = ". {} :\n"
PARAM = "  ├── {} : {}\n"
PARAMSINGLE = "  ├── {}\n"
LPARAM = "  └── {} : {}\n\n"
LPARAMSINGLE = "  └── {}\n\n"
TITLE = "| {} |\n"

# % CLASS


class InfoString(object):
    """Generates info strings

    Parameters
    ----------
    title : str
        the title of the infostring object
    """

    def __init__(self, title: str):
        self.__title = title
        self.__elements = OrderedDict()
        self.__current_section = ""

    @property
    def elements(self):
        """OrderedDict: the elements inside the infostring object"""
        return self.__elements

    @property
    def title(self) -> str:
        """str: titile of the infostring object"""
        return self.title

    @title.setter
    def title(self, value: str):
        assert isinstance(value, str), "title should be a string"
        self.__title = value

    def add_section(self, name: str):
        """adds a new section for parameters

        Parameters
        ----------
        name : str
            the section name
        """
        if name in self.__elements:
            raise Warning(f"section '{name}' already exists")
        self.__elements[name] = OrderedDict()
        self.__current_section = name

    def add_element(self, name: str, value: str = None, section: str = None):
        """adds an element in a given section

        Parameters
        ----------
        name : str
            name of the element
        value : str, optional
            value of the element, by default None
        section : str, optional
            name of the section. If None is given, the element is added to
            the lastly used section, by default None
        """
        if section is None:
            section = self.__current_section
        if section not in self.__elements:
            raise Warning(f"section '{section}' does not exist")
        # switch current section
        self.__current_section = section
        if name in self.__elements[section]:
            raise Warning(f"section '{section}' already has an element {name}")
        self.__elements[section][name] = value

    def rm_element(self, name: str, section: str = None):
        """removes an element from a section

        Parameters
        ----------
        name : str
            name of the element to remove
        section : str, optional
            name of the section. If None is given, the element is removed from
            the lastly used section, by default None
        """

        if section is None:
            section = self.__current_section
        if section not in self.__elements:
            raise Warning(f"section '{section}' does not exist")
        # switch current section
        self.__current_section = section
        if name not in self.__elements[section]:
            raise Warning(f"section '{section}' does not have an element {name}")
        del self.__elements[section][name]

    def absorb_section(self, info, target_section: str, new_name: str = None):
        """incorporates a section from another infostring object

        Parameters
        ----------
        info : InfoString
            the infostring object from which we take the section
        target_section : str
            name of the section to incorporate
        new_name : str, optional
            name of the incorporated section.
            If None is given we use the name of the original section, by default None
        """
        # assert
        assert isinstance(info, InfoString), "'info' should be an InfoString object"
        # get info
        info_dic = info.elements
        assert (
            target_section in info_dic
        ), f"This info object does not have a '{target_section}' section"
        if new_name is None:
            new_name = target_section
        # absorb
        self.add_section(new_name)
        for name, value in info_dic[target_section].items():
            self.add_element(name, value)

    def merge(self, info, prefix=""):
        """merges an entire infostring

        Parameters
        ----------
        info : InfoString
            infostring object to merge
        prefix : str, optional
            prefix added to the names of the sections incorporated
            from the merged infostring, by default ""
        """
        # assert
        assert isinstance(info, InfoString), "'info' should be an InfoString object"
        # merge
        for section, elements in info.elements.items():
            self.add_section(prefix + section)
            for name, value in elements.items():
                self.add_element(name, value)

    def generate(self, display_title=True):
        """generates a string from the infostring object

        Parameters
        ----------
        display_title : bool, optional
            whether to display the title, by default True

        Returns
        -------
        out_str: str
            a string with all parameters from the info string
        """
        # init
        out = []
        # title
        if display_title:
            title = TITLE.format(self.__title)
            line = "─" * (len(title) - 1) + "\n"
            out.append(line + title + line)
        # params
        for section, elements in self.__elements.items():
            out.append(HEADER.format(section))
            for name, value in elements.items():
                if value is None:
                    out.append(PARAMSINGLE.format(name))
                else:
                    out.append(PARAM.format(name, value))
            # remove last an replace by a last param string
            out.pop()
            if value is None:
                out.append(LPARAMSINGLE.format(name))
            else:
                out.append(LPARAM.format(name, value))

        out_str = "".join(out)
        return out_str
