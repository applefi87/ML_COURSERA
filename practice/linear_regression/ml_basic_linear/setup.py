from setuptools import setup, find_packages

setup(
    name="ml_basic_linear",
    version="0.1",
    package_dir={"": "src"},  # This tells setuptools that packages are under src directory
    packages=find_packages(where="src"),  # This will find packages in src directory
)