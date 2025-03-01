#===================================================================================================
# Monster Piano Transformer sample_midis Python module
#===================================================================================================
# Project Los Angeles
# Tegridy Code 2025
#===================================================================================================
# License: Apache 2.0
#===================================================================================================
'''
import importlib.resources as pkg_resources
from monsterpianotransformer import seed_midis

#===================================================================================================

def get_sample_midi_files():
    
    midi_files = []
    
    for resource in pkg_resources.contents(seed_midis):
        if resource.endswith('.mid'):
            with pkg_resources.path(seed_midis, resource) as p:
                midi_files.append((resource, str(p)))
                
    return sorted(midi_files)
'''

import os
from pathlib import Path

def get_sample_midi_files():
    midi_files = []
    seed_dir = Path('./seed_midis')
    
    if not seed_dir.exists():
        raise FileNotFoundError(f"Directory {seed_dir} not found")
        
    for resource in seed_dir.iterdir():
        if resource.suffix == '.mid':
            midi_files.append((resource.name, str(resource.resolve())))
            
    return sorted(midi_files)

#===================================================================================================
# This is the end of sample_midis Python module
#===================================================================================================