from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension


extensions = [Extension('crdfgen',
                        ['MDutils/crdfgen.pyx'])]

setup(
    name='MDutils',
    version='v1.0.0',
    packages=['MDutils'],
    ext_modules=cythonize(extensions, language_level="3", annotate=True),
    url='https://github.com/t-young31/MDutils',
    entry_points={'console_scripts': ['rdfgen = MDutils.rdfgen:main']},
    license='MIT',
    author='Tom Young',
    author_email='tom.young@chem.ox.ac.uk',
    description='Python module for MD analysis'
)
