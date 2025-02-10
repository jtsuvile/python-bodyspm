import setuptools
setuptools.setup(     
     name="sub-module",     
     version="0.0.1",
     python_requires=">=3.6",   
     packages=setuptools.find_packages(exclude=('tests', 'docs')),
)