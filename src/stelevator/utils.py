import os
from collections import UserList

_PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
_DATADIR = os.path.join(_PACKAGEDIR, 'data')

# def replace_docstring(oldvalue, newvalue):
#     """Replace 'oldvalue' with 'newvalue' in the docstring of decorated function."""
#     def decorator(func):
#         def wrapper(*args, **kwargs):
#             return func(*args, **kwargs)
#         wrapper.__doc__ = func.__doc__.replace(oldvalue, newvalue)
#         return wrapper
#     return decorator
