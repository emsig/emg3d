import time
from emg3d import __version__

# ==== 1. Extensions  ====

# Load extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.intersphinx',
    'numpydoc',
    'sphinx_automodapi.automodapi',
]
autosummary_generate = True
add_module_names = True

# Numpydoc settings
numpydoc_show_class_members = False
numfig = True
numfig_format = {'figure': 'Figure %s:'}

# Todo settings
todo_include_todos = True

# Intersphinx configuration
intersphinx_mapping = {
    "numpy": ("https://docs.scipy.org/doc/numpy", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "discretize": ("https://discretize.simpeg.xyz/en/master", None),
    "empymod": ("https://empymod.readthedocs.io/en/stable", None),
    "xarray": ("https://xarray.pydata.org/en/stable", None),
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
copyright = u'2018-{}, The emg3d Developers.'.format(time.strftime("%Y"))
author = 'The emg3d Developers'

# |version| and |today| tags (|release|-tag is not used).
version = __version__
release = __version__
today_fmt = '%d %B %Y'

# List of patterns to ignore, relative to source directory.
exclude_patterns = ['_build', '../tests']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'friendly'

# ==== 3. HTML settings ====
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'logo_only': True,
    'display_version': True,
    'prev_next_buttons_location': 'both',
}
html_static_path = ['_static']
html_logo = '_static/logo-emg3d-cut.svg'
html_favicon = '_static/favicon.ico'
html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'searchbox.html',
    ]
}

html_context = {
    'menu_links_name': 'Links',
    'menu_links': [
        ('<i class="fa fa-link fa-fw"></i> Website',
         'https://empymod.github.io'),
        ('<i class="fa fa-github fa-fw"></i> Source Code',
         'https://github.com/empymod/emg3d'),
    ],
}

htmlhelp_basename = 'emg3ddoc'


# -- CSS fixes --
def setup(app):
    app.add_css_file("style.css")


# ==== 4. Other Document Type Settings ====
# Options for LaTeX output
latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '10pt',
}
latex_documents = [
    (master_doc, 'emg3d.tex', 'emg3d Documentation',
     'The emg3d Developers', 'manual'),
]

# Options for manual page output
man_pages = [
    (master_doc, 'emg3d', 'emg3d Documentation',
     [author], 1)
]

# Options for Texinfo output
texinfo_documents = [
    (master_doc, 'emg3d', 'emg3d Documentation',
     author, 'emg3d', description,
     'Electromagnetic geophysical modelling'),
]


# ==== 5. linkcheck ====

# Papers from academic.oup results in a 104 error
linkcheck_ignore = [
    'https://doi.org/10.1111/j.1365-246X.2010.04544.x',
    'https://doi.org/10.1088/0266-5611/24/3/034012',
]
