# -*- coding: utf-8 -*-
#
# YATSM documentation build configuration file, created by
# sphinx-quickstart on Tue Nov  4 18:26:04 2014.
import sys
import os

import sphinx

try:
    from unittest import mock
    from unittest.mock import MagicMock
except:
    import mock
    from mock import MagicMock
mock.FILTER_DIR = False


MOCK_MODULES = [
    'glmnet',
    'matplotlib', 'matplotlib.pyplot', 'matplotlib.style',
    'numpy', 'numpy.lib', 'numpy.lib.recfunctions', 'numpy.ma',
    'numba',
    'osgeo',
    'palettable',
    'pandas',
    'patsy',
    'rpy2', 'rpy2.robjects', 'rpy2.robjects.numpy2ri',
    'rpy2.robjects.packages',
    'scipy', 'scipy.stats',
    'sklearn', 'sklearn.cross_validation', 'sklearn.ensemble',
    'sklearn.linear_model',
    'sklearn.externals', 'sklearn.externals.joblib',
    'statsmodels', 'statsmodels.api',
    'yatsm.accel', 'yatsm._cyprep',
    'yatsm.classifiers', 'yatsm.classifiers.diagnostics',
]
for mod_name in MOCK_MODULES:
    print 'Mocking %s' % mod_name
    sys.modules[mod_name] = MagicMock()

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
d = os.path.dirname
sys.path.insert(0, d(d(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(d(d(os.path.abspath(__file__))), 'scripts'))

import yatsm

# Add scripts directory to PATH for sphinxcontrib.programoutput
os.environ['PATH'] = '{root}{sep}{dir}{psep}{path}'.format(
    root=d(d(__file__)), sep=os.sep, dir='scripts',
    psep=os.pathsep, path=os.environ['PATH'])

# -- General configuration ------------------------------------------------
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.graphviz',
    'sphinx.ext.todo',
    'sphinxcontrib.programoutput',
    'sphinxcontrib.bibtex'
]
# Napoleon extension moving to sphinx.ext.napoleon as of sphinx 1.3
sphinx_version = sphinx.version_info
if sphinx_version[0] >= 1 and sphinx_version[1] >= 3:
    extensions.append('sphinx.ext.napoleon')
else:
    extensions.append('sphinxcontrib.napoleon')

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'YATSM'
copyright = u'2014 - 2015, Chris Holden'

version = yatsm.__version__
release = yatsm.__version__
html_last_updated_fmt = '%c'

exclude_patterns = ['_build']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []

# -- Options for HTML output ----------------------------------------------

# on_rtd is whether we are on readthedocs.org, this line of code grabbed from
# docs.readthedocs.org
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if not on_rtd:  # only import and set the theme if we're building docs locally
    import sphinx_rtd_theme
    html_theme = 'sphinx_rtd_theme'
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_context = dict(
    display_github=True,
    github_user="ceholden",
    github_repo="yatsm",
    github_version="master",
    conf_py_path="/docs/"
)

# html_theme_options = { }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Add any extra paths that contain custom files (such as robots.txt or
# .htaccess) here, relative to this directory. These files are copied
# directly to the root of the documentation.
# html_extra_path = []

# Custom sidebar templates, maps document names to template names.
# html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
# html_additional_pages = {}

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
# html_use_opensearch = ''
