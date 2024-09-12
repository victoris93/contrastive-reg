from setuptools import setup, find_packages
import os


# +
# with open('README.md', 'r') as f:
#     long_description = f.read()
# -

with open('requirements.txt', 'r') as f:
    requirements = f.read().split("\n")

setup(
    name='ContModeling',
    version='0.0.0',
#     description='A BIDS toolbox for connectivity & gradient analyses.',
#     long_description=long_description,
#     long_description_content_type='text/markdown',
    author='Malo Renaudin',
    author_email='malo.renaudin@hec.edu',
    python_requires='>=3.6',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=requirements,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ]
)
