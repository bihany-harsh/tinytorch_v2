# the setup.py file is written by Claude Sonnet-4.5 (Unextended) (as of this commit)
import os
import sys
import shutil
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        import subprocess
        
        # This is where setuptools expects the output
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # CMake configure arguments
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            '-DCMAKE_BUILD_TYPE=Release'
        ]
        
        # Build directory
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        
        # Configure
        subprocess.check_call(
            ['cmake', ext.sourcedir] + cmake_args,
            cwd=self.build_temp
        )
        
        # Build
        subprocess.check_call(
            ['cmake', '--build', '.', '--config', 'Release'],
            cwd=self.build_temp
        )

setup(
    name='tinytorch',
    version='0.1.0',
    packages=find_packages(),
    ext_modules=[CMakeExtension('tinytorch._core')],
    cmdclass={'build_ext': CMakeBuild},
    python_requires='>=3.7',
)