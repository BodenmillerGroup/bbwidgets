from setuptools import setup

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='bbwidgets',
    version='0.1.0',
    packages=['bbwidgets'],
    url='https://github.com/BodenmillerGroup/bbwidgets',
    license='MIT',
    author='Jonas Windhager',
    author_email='jonas.windhager@uzh.ch',
    description='Interactive Widgets for the Jupyter Notebook',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=['ipywidgets', 'matplotlib', 'numpy', 'traitlets', 'traittypes'],
    classifiers=[
        'Framework :: Jupyter',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',

    ]
)
