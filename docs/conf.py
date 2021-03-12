import time
from emg3d import __version__

# ==== 1. Extensions  ====

# Load extensions
extensions = [
    'sphinx.ext.autodoc',
    'numpydoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.doctest',
    # 'sphinx.ext.todo',
    'sphinx_automodapi.automodapi',
    'matplotlib.sphinxext.plot_directive',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
]
autosummary_generate = True
add_module_names = True
add_function_parentheses = False

# Numpydoc settings
numpydoc_show_class_members = False
numfig = True
# Make numpydoc to generate plots for example sections
numpydoc_use_plots = True

# Todo settings
todo_include_todos = True

# Intersphinx configuration
intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "discretize": ("https://discretize.simpeg.xyz/en/master", None),
    "empymod": ("https://empymod.readthedocs.io/en/stable", None),
    "xarray": ("https://xarray.pydata.org/en/stable", None),
    "numba": ("https://numba.readthedocs.io/en/stable", None),
}

# ==== 2. General Settings ====
description = 'A multigrid solver for 3D electromagnetic diffusion.'

# The templates path.
templates_path = ['_templates']

# The suffix(es) of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'emg3d'
author = 'The emg3d Developers'
copyright = f'2018-{time.strftime("%Y")}, {author}'

# |version| and |today| tags (|release|-tag is not used).
version = __version__
release = __version__
today_fmt = '%d %B %Y'

# List of patterns to ignore, relative to source directory.
exclude_patterns = ['_build', '../tests']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'friendly'

# ==== 3. HTML settings ====
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_logo = '_static/logo-emg3d-cut.svg'
html_favicon = '_static/favicon.ico'

html_theme_options = {
  "github_url": "https://github.com/emsig/emg3d",
  "external_links": [
      {"name": "EMSiG", "url": "https://emsig.github.io"},
  ],
  # "use_edit_page_button": True,
}

html_context = {
    "github_user": "emsig",
    "github_repo": "emg3d",
    "github_version": "master",
    "doc_path": "docs",
}

html_use_modindex = True
html_file_suffix = '.html'
htmlhelp_basename = 'emg3d'

# ==== 4. linkcheck ====

# Papers from academic.oup results in a 104 error
linkcheck_ignore = [
    'https://doi.org/10.1111/j.1365-246X.2010.04544.x',
    'https://doi.org/10.1088/0266-5611/24/3/034012',
]
