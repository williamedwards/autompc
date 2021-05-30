from setuptools import setup, find_packages

setup(
        name="autompc",
        version="0.0.1",
        description="A library for the creation and tuning of System ID+MPC pipelines",
        author="William Edwards",
        author_email="williamedwards314@gmail.com",
        url="https://github.com/williamedwards/autompc",
        packages=find_packages(include=["autompc", "autompc.*"])
)
