# import setup function from 
# python distribution utilities
from setuptools import setup, find_packages
from distutils.core import setup 

with open('README.md') as f:
    README = f.read()
  
# Calling the setup function
setup(
      name = 'jitterbug',
      version = '1.0.0',
      # py_modules = ['addition'],
      license="MIT",
      packages=find_packages(),
      install_requires=['numpy', 'pandas', 'scipy', 'bayesian-changepoint-detection'],
      python_requires='>=3',
      author ='Esteban Carisimo',
      author_email = 'esteban.carisimo@northwestern.edu',
      url = 'https://www.github.com/estcarisimo/jitterbug',
      description = 'Framework for jitter-based network congestion inference',
      long_description=README,
      keywords='jitter, RTT measurements',
      classifiers=[
            # Trove classifiers
            # (https://pypi.python.org/pypi?%3Aaction=list_classifiers)
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Topic :: System :: Networking :: Monitoring',
            'Intended Audience :: Science/Research',
      ],
      entry_points={'console_scripts': ['jitterbug=tools.jitterbug:main',],}
)