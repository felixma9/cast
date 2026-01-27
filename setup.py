"""
Setup script for building strain_utility PyBind11 extension module.

Build with: python setup.py build_ext --inplace
Or: pip install -e .
"""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import pybind11
import os
import struct

# Force 64-bit build on Windows
if sys.platform == 'win32':
    os.environ['DISTUTILS_USE_SDK'] = '1'
    # Update PATH to use x64 compiler
    msvc_path = r"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\HostX64\x64"
    os.environ['PATH'] = msvc_path + os.pathsep + os.environ.get('PATH', '')

# Custom build_ext to handle 64-bit library paths
class build_ext_64bit(build_ext):
    def build_extensions(self):
        # Detect if we're building for 64-bit Python
        if struct.calcsize("P") == 8:  # 64-bit
            # Override the default library directories with 64-bit paths
            for ext in self.extensions:
                # Remove default 32-bit paths
                ext.library_dirs = [
                    r"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\lib\x64",
                    r"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\ATLMFC\lib\x64",
                    r"C:\Program Files (x86)\Windows Kits\10\lib\10.0.26100.0\ucrt\x64",
                    r"C:\Program Files (x86)\Windows Kits\10\lib\10.0.26100.0\um\x64",
                ]
        super().build_extensions()


class get_pybind11_include:
    def __str__(self):
        return pybind11.get_include()

ext_modules = [
    Extension(
        'strain_utility',
        [
            'cpp/strain_utility_bindings.cpp',
            'cpp/StrainUtility.cpp',
        ],
        include_dirs=[
            str(get_pybind11_include()),
            'cpp',
        ],
        language='c++',
        extra_compile_args=['/std:c++17'] if sys.platform == 'win32' else ['-std=c++17'],
    ),
]

setup(
    name='strain_utility',
    version='1.0.0',
    author='Your Name',
    description='PyBind11 bindings for StrainUtility C++ library',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.6.0'],
    cmdclass={'build_ext': build_ext_64bit},
    zip_safe=False,
)
