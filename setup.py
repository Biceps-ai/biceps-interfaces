from setuptools import setup


with open("requirements.txt", "r") as fp:
    requirements = [line.strip() for line in fp.readlines()]


setup(
    name='biceps_interfaces',
    version='0.0.1',
    authors = [
        {'name': "Niels Pichon", 'email': "niels@biceps.ai"},
    ],
    install_requires=requirements,
)
