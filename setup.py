from setuptools import setup

setup(
    name='modesto',
    version='0.1',
    packages=['modesto'],
    package_dir={'modesto': 'modesto'},
    package_data={'modesto': ['Data/PipeCatalog/*.txt', 'Data/RenewableProduction/*.txt', 'Data/Weather/*.txt']},
    url='',
    license='',
    author='Annelies Vandermeulen and Bram van der Heijde',
    author_email='',
    description='Multi-Objective District Energy Systems Toolbox for Optimization', install_requires=['pyomo', 'pandas']
)
