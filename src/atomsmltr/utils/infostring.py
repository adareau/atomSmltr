# -*- coding: utf-8 -*-
"""tools to generate and display info strings
"""
# % IMPORTS

from collections import OrderedDict

# % CONSTANTS

HEADER = ". {} :\n"
PARAM = "  ├── {} : {}\n"
LPARAM = "  └── {} : {}\n\n"
TITLE = "| {} |\n"

# % CLASS


class InfoString(object):
    """Allows to generate info strings"""

    def __init__(self, title: str):
        self.__title = title
        self.__elements = OrderedDict()
        self.__current_section = ""

    def add_section(self, name: str):
        if name in self.__elements:
            raise Warning(f"section '{name}' already exists")
        self.__elements[name] = OrderedDict()
        self.__current_section = name

    def add_element(self, name: str, value: str, section=None):
        if section is None:
            section = self.__current_section
        if section not in self.__elements:
            raise Warning(f"section '{section}' does not exist")
        # switch current section
        self.__current_section = section
        if name in self.__elements[section]:
            raise Warning(f"section '{section}' already has an element {name}")
        self.__elements[section][name] = value

    def generate(self):
        # init
        out = []
        # title
        title = TITLE.format(self.__title)
        line = "─" * (len(title) - 1) + "\n"
        out.append(line + title + line)
        # params
        for section, elements in self.__elements.items():
            out.append(HEADER.format(section))
            for name, value in elements.items():
                out.append(PARAM.format(name, value))
            # remove last an replace by a last param string
            out.pop()
            out.append(LPARAM.format(name, value))

        out_str = "".join(out)
        return out_str
