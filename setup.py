from setuptools import setup, find_packages
import os


setup(
    name="retrievepy",
    version="0.0.3",
    author="Matthew Hanley",
    author_email="matthew.hanley@lasp.colorado.edu",
    description=("Spacecraft data retrieval using the LaTiS API from LASP"),
    license="MIT",
    keywords="data retrieval latis",
    url="https://github.com/matthewdhanley/retrievepy",
    packages=find_packages(),
    data_files=[('', ['pylogger/logger.cfg']), ('', ['retrievepy/latis.cfg'])],
)