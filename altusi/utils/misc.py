"""
Misc Lib
========

Miscellaneous functions to support programming
"""


import os

def get_fname(file_path):
    _, full_name = os.path.split(file_path)
    fname, ext = os.path.splitext(full_name)

    return fname, ext
