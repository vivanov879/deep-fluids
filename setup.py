from setuptools import find_namespace_packages, setup

setup(
    name='deep-fluids',
    version="0.0.1",
    package_dir={"": "src"},
    packages=find_namespace_packages("src"),
    url='',
    license='',
    author='vivanov',
    author_email='vivanov879@ya.ru',
    description='Torch implementation of deep fluids paper',
)
