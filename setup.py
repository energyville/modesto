from setuptools import setup, find_packages

setup(
    name='modesto',
    version='0.2.1',
    packages=find_packages(),
    package_dir={'modesto': 'modesto'},
    package_data={'': ['*.txt', '*.rst', '*.csv']},
    setup_requires=['pytest-runner<3'],
    tests_require=['pytest'],
    url='',
    license='',
    author='Annelies Vandermeulen and Bram van der Heijde',
    author_email='',
    description='Multi-Objective District Energy Systems Toolbox for Optimization',
    install_requires=['pyomo', 'pandas', 'networkx>=2.0', 'numpy', 'setuptools-git'],
    include_package_data=True
)
