# python setup.py develop
from setuptools import setup


CLASSIFIERS = '''\
License :: OSI Approved
Programming Language :: Python :: 3.6.10
"Intended Audience :: Science/Research",
"Topic :: Scientific/Engineering",
"Topic :: Scientific/Engineering :: Mathematics",
"Operating System :: OS Independent"
'''

DISTNAME = 'pytau'
AUTHOR = 'Abuzar Mahmood'
AUTHOR_EMAIL = 'abuzarmahmood@gmail.com'
DESCRIPTION = 'Simple package to streamline batch inference on pymc3-based changepoint models.'
LICENSE = 'MIT'
README = 'Streamlined batch inference on changepoint models'

VERSION = '0.1.0'
ISRELEASED = False

PYTHON_MIN_VERSION = '3.6.10'
PYTHON_MAX_VERSION = '3.6.10'
PYTHON_REQUIRES = f'>={PYTHON_MIN_VERSION}, <={PYTHON_MAX_VERSION}'

INSTALL_REQUIRES = [
    'pymc3'
]

PACKAGES = [
    'pytay',
]

metadata = dict(
    name=DISTNAME,
    version=VERSION,
    long_description=README,
    packages=PACKAGES,
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    classifiers=[CLASSIFIERS],
    license=LICENSE
)


def setup_package() -> None:
    setup(**metadata)


if __name__ == '__main__':
    setup_package()
