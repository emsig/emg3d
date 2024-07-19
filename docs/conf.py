import time
from emg3d import __version__

# ==== 1. Extensions  ====

# Load extensions
extensions = [
    # 'sphinx.ext.autodoc',
    'numpydoc',
    'sphinx_design',
    'sphinx.ext.intersphinx',
    # 'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    # 'sphinx.ext.doctest',
    'sphinx.ext.todo',
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
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "discretize": ("https://discretize.simpeg.xyz/en/main", None),
    "empymod": ("https://empymod.emsig.xyz/en/stable", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),
    "numba": ("https://numba.readthedocs.io/en/stable", None),
}

# ==== 2. General Settings ====
description = 'A multigrid solver for 3D electromagnetic diffusion.'

# The templates path.
# templates_path = ['_templates']

# The suffix(es) of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'emg3d'
author = 'The emsig community'
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
html_logo = '_static/emg3d-logo.svg'
html_favicon = '_static/favicon.ico'

html_theme_options = {
    "github_url": "https://github.com/emsig/emg3d",
    "external_links": [
        {"name": "Gallery", "url": "https://emsig.xyz/emg3d-gallery/gallery"},
        {"name": "emsig", "url": "https://emsig.xyz"},
    ],
    'navigation_with_keys': True,
    # "use_edit_page_button": True,
}

html_context = {
    "github_user": "emsig",
    "github_repo": "emg3d",
    "github_version": "main",
    "doc_path": "docs",
}

html_use_modindex = True
html_file_suffix = '.html'
htmlhelp_basename = 'emg3d'
html_css_files = [
    "style.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/" +
    "css/font-awesome.min.css"
]

# ==== 4. linkcheck ====

# Many journals do not allow the ping; no idea why. I exclude all DOI's from
# the check; they "should" be permanent, that is their entire purpose; doi.org
# is responsible for resolving them.
linkcheck_ignore = [
    "https://doi.org/*",  # DOIs should be permanent (their entire purpose)
]
