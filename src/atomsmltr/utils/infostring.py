# -*- coding: utf-8 -*-
"""tools to generate and display info strings
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
    """Allows to generate info strings"""

    def __init__(self, title: str):
        self.__title = title
        self.__elements = OrderedDict()
        self.__current_section = ""

    @property
    def elements(self):
        return self.__elements

    def add_section(self, name: str):
        if name in self.__elements:
            raise Warning(f"section '{name}' already exists")
        self.__elements[name] = OrderedDict()
        self.__current_section = name

    def add_element(self, name: str, value: str = None, section=None):
        if section is None:
            section = self.__current_section
        if section not in self.__elements:
            raise Warning(f"section '{section}' does not exist")
        # switch current section
        self.__current_section = section
        if name in self.__elements[section]:
            raise Warning(f"section '{section}' already has an element {name}")
        self.__elements[section][name] = value

    def absorb_section(self, info, target_section, new_name=None):
        """incorporates the section 'section' from a info object 'info'"""
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
        """Merges info strings"""
        # assert
        assert isinstance(info, InfoString), "'info' should be an InfoString object"
        # merge
        for section, elements in info.elements.items():
            self.add_section(prefix + section)
            for name, value in elements.items():
                self.add_element(name, value)

    def generate(self, display_title=True):
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
