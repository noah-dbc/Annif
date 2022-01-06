import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='annif',
    version='0.56.0-dev',
    url='https://github.com/NatLibFi/Annif',
    author='Osma Suominen',
    author_email='osma.suominen@helsinki.fi',
    description='Automated subject indexing and classification tool',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.7',
    install_requires=[
        'connexion[swagger-ui]',
        'swagger_ui_bundle',
        'flask>=1.0.4,<2',
        'flask-cors',
        'click==7.1.*',
        'click-log',
        'joblib==1.0.1',
        'nltk',
        'gensim==4.1.*',
        'scikit-learn==0.24.2',
        'scipy==1.5.4',
        'rdflib>=4.2,<7.0',
        'gunicorn',
        'numpy==1.19.*',
        'optuna==2.8.0',
        'stwfsapy==0.3.*',
        'python-dateutil',
    ],
    tests_require=['py', 'pytest', 'requests'],
    extras_require={
        'fasttext': ['fasttext==0.9.2'],
        'voikko': ['voikko'],
        'nn': ['tensorflow-cpu==2.5.0', 'lmdb==1.2.1'],
        'omikuji': ['omikuji==0.4.*'],
        'yake': ['yake==0.4.5'],
        'pycld3': ['pycld3'],
        'dev': [
            'codecov',
            'pytest-cov',
            'pytest-watch',
            'pytest-flask',
            'pytest-flake8',
            'bumpversion',
            'autopep8'
        ]
    },
    entry_points={
        'console_scripts': ['annif=annif.cli:cli']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ]
)
