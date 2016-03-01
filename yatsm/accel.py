""" Decorator ``try_jit`` accelerates computation via Numba, when available
"""
from functools import wraps
import logging
import os

logger = logging.getLogger('yatsm')

has_numba = True
try:
    import numba as nb
except ImportError:
    has_numba = False
else:
    # Check if disabled via environment
    if os.environ.get('NUMBA_DISABLE_JIT') == '1':
        has_numba = False
    # Check for an adequate version number
    try:
        nb_ver = nb.__version__.split('.')
        if int(nb_ver[0]) == 0 and int(nb_ver[1]) < 22:
            has_numba = False
            logger.warning('You have numba installed, but the version is too '
                           'old to be used (%s)' % nb.__version__)
    except:
        logger.warning('Could not parse numba version. Disabling numba.jit')
        has_numba = False


def _doublewrap(f):
    """ Allows decorators to be called with/without args/kwargs

    Allows:

    .. code-block:: python

        @decorator
        @decorator()
        @decorator(args, kwargs=values)

    Modification of answer from user "bj0" on StackOverflow:
    http://stackoverflow.com/a/14412901
    """
    @wraps(f)
    def new_dec(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            # called as @decorator
            return f(args[0])
        elif len(args) == 1 and callable(args[0]):
            # called as decorator(f, **kwargs)
            return f(args[0], **kwargs)
        elif len(args) == 0 and len(kwargs) == 0:
            # called as @decorator()
            return f
        else:
            # called as @decorator(*args, **kwargs)
            return lambda realf: f(realf, *args, **kwargs)

    return new_dec


@_doublewrap
def try_jit(f, *args, **kwargs):
    """ Apply numba.jit to function ``f`` if Numba is available

    Accepts arguments to Numba jit function (signature, nopython, etc.).

    Examples:

    .. code-block:: python

        @try_jit
        @try_jit()
        @try_jit(nopython=True)
        @try_jit("float32[:](float32[:], float32[:])", nopython=True)

    """
    if has_numba:
        @wraps(f)
        def wrap(*args, **kwargs):
            return nb.jit(*args, **kwargs)
        return wrap(f, *args, **kwargs)
    else:
        return f
