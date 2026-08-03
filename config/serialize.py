"""Dump every Train config in a python config module to YAML/JSON.

Usage: python -m config.serialize config.cheetah [outdir]
"""
import importlib
import sys
from pathlib import Path

from codesign.config import Train


def serialize_module(module_name, outdir='config/serialization'):
    """Write each module-level Train as `<outdir>/<varname>.{yaml,json}`.
    Returns the list of Paths written."""
    module = importlib.import_module(module_name)
    written = []
    for name, obj in vars(module).items():
        if isinstance(obj, Train):
            stem = Path(outdir) / name
            written.append(obj.save_yaml_path(stem.with_suffix('.yaml')))
            written.append(obj.save_json_path(stem.with_suffix('.json')))
    return written


if __name__ == '__main__':
    for path in serialize_module(*sys.argv[1:]):
        print(path)
