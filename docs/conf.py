import sys
from pathlib import Path

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "OpenEquivariance"
copyright = "2025, The Regents of the University of California, through Lawrence Berkeley National Laboratory."
author = "Vivek Bharadwaj, Austin Glover, Aydin Buluc, James Demmel"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
# html_static_path = ["_static"]

extensions = [
    "sphinx.ext.autodoc",
]

sys.path.insert(0, str(Path("..").resolve()))

autodoc_mock_imports = ["torch", "openequivariance.extlib", "jinja2", "numpy"]
autodoc_typehints = "description"
