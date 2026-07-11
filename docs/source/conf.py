# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os, sys
sys.path.insert(0, os.path.abspath('../../src'))

project = 'COSMIX'
copyright = '2026, Ameya Kolhatkar'
author = 'Ameya Kolhatkar'
release = '1.3.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',     # Pulls docstrings from your python code
    'sphinx.ext.napoleon',    # Parses Google/NumPy-style docstrings cleanly
    'myst_parser'             # Allows you to write docs in Markdown (.md)
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
