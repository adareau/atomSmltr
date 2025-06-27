# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Python imports -----------------------------------------------------
import os
import sys

# Location of Sphinx files
sys.path.insert(0, os.path.abspath("./../"))  ##Add the folder one level above
os.environ["SPHINX_APIDOC_OPTIONS"] = (
    "members,show-inheritance"  ## Hide undocumented members
)
import sphinx.ext.apidoc


def setup(app):
    app.add_css_file("css/stylesheet.css")
    app.add_js_file("webcode/summaryOpen.js")
    sphinx.ext.apidoc.main(
        [
            "-f",  # Overwrite existing files
            "-T",  # Create table of contents
            "-e",  # Give modules their own pages
            "-E",  # user docstring headers
            "-M",  # Modules first
            "-o",  # Output the files to:
            "./_autogen/",  # Output Directory
            "./../src/atomsmltr",  # Main Module directory
        ]
    )


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "atomSmltr"
copyright = "2025, A. Dareau"
author = "A. Dareau"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    # "sphinx.ext.ifconfig",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_copybutton",
    "myst_nb",
    "sphinx_thebe",
    "sphinx_favicon",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "restructuredtext",
    ".ipynb": "myst-nb",
}

suppress_warnings = [
    "docutils",  # Suppress the anoying "Inline substitution_reference start-string without end-string"
]


# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# --- myst options
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_logo = "./_static/images/atomsmltr_logo.svg"
html_favicon = "./_static/images/favicon.ico"
html_static_path = ["_static"]
html_theme_options = {
    # "announcement": announcement,
    "logo": {
        "text": "atomSmltr",
    },
    "header_links_before_dropdown": 4,
    "show_version_warning_banner": True,
    "navbar_align": "content",  # [left, content, right] For testing that the navbar items align properly
    "navbar_center": ["navbar-nav"],
    "navbar_persistent": ["version-switcher"],
    "check_switcher": True,
    "navigation_with_keys": False,
    "footer_start": ["copyright"],
    "footer_end": [],
    "use_edit_page_button": True,
    "navigation_depth": 3,
    "collapse_navigation": False,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/adareau/atomSmltr",
            "icon": "https://img.shields.io/github/stars/adareau/atomSmltr?style=social",
            "type": "url",
        },
    ],
}
html_context = {
    "github_user": "adareau",
    "github_repo": "atomsmltr",
    "github_version": "main",
    "doc_path": "docs/",
}
html_sidebars = {
    "**": ["search-field.html", "sidebar-nav-bs.html", "sidebar-ethical-ads.html"],
}

# --- switcher
json_url = "https://atomsmltr.readthedocs.io/en/latest/_static/switcher.json"
