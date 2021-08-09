# module imports
import re
import os
import os.path as op
from setuptools import setup, find_packages

# sets the readme file object
curdir = op.dirname(op.realpath(__file__))
readme = open(op.join(curdir, 'README.md')).read()

requirements = [
    "numpy <= 1.19.1",
    "pycparser <= 2.20",
    "pyopencl <= 2020.2",
    "pyOpenSSL <= 19.1.0",
    "PySocks <= 1.7.1",
    "pytools <= 2020.3.1",
    "pywin32 <= 227",
    "requests <= 2.23.0",
    "ruamel-yaml <= 0.15.87",
    "six <= 1.14.0",
    "urllib3 <= 1.25.8",
    "win-inet-pton <= 1.1.0",
    "wincertstore <= 0.2",
]

setup(
    name='EPhyAnalysis',
    version='1.0',
    description='GUI-based program that analyses the spiking activity from Neuropixel recording experiment',
    long_description=readme,
    packages=find_packages(),
    include_package_data=True,
    install_requires = [
        # ''               # margrie_libs
        # 'https://github.com/stephenlenzi/probez.git',      # probez repository
        # ''               # pyphys
        # ''               # rotation_analysis
        # ''               # vest_phys
        requirements,
    ],
    url="https://github.com/RichardFav/AnalysisGUI",
    author="Richard Faville",
    author_email="richard.faville@gmail.com",
)


# from setuptools import setup, find_packages
#
# requirements = [
#     "numpy",
#     "scipy <= 1.2.1",
#     "pandas",
#     "matplotlib",
#     "seaborn",
#     "pycircstat",
#     "nose",
#     "decorator",
#     "xlrd",
# ]
#
#
# setup(
#     name="opendirection",
#     version="0.1.0",
#     description="Analysis of spiking activity in an open field",
#     packages=find_packages(),
#     include_package_data=True,
#     install_requires=requirements,
#     entry_points={
#         "console_scripts": [
#             "opendirection = opendirection.main:main",
#             "opendirection_batch = opendirection.batch:main",
#             "gen_velo_profile = opendirection.utils.generate_velo_profile:main",
#         ]
#     },
#     url="https://github.com/adamltyson/opendirection",
#     author="Adam Tyson",
#     author_email="adam.tyson@ucl.ac.uk",
# )


# setup(
#     name = 'MyProject',
#     version = '0.1.0',
#     url = '',
#     description = '',
#     packages = find_packages(),
#     install_requires = [
#         # Github Private Repository
#         'ExampleRepo @ git+ssh://git@github.com/example_organization/ExampleRepo.git#egg=ExampleRepo-0.1'
#     ]
# )
#
# setup(
#     name='<package>',
# ...
#     install_requires=[
#         '<normal_dependency>',
#          # Private repository
#         '<dependency_name> @ git+ssh://git@github.com/<user>/<repo_name>@<branch>',
#          # Public repository
#         '<dependency_name> @ git+https://github.com/<user>/<repo_name>@<branch>',
#     ],
# )
