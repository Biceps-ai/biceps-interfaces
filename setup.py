from setuptools import setup


setup(
    name='biceps-interfaces',
    version='0.0.2',
    authors = [
        {'name': "Niels Pichon", 'email': "niels@biceps.ai"},
    ],
    python_requires='>=3.10,<3.12',
    install_requires=[
        "dataclasses-json>=0.6.3",
        "torch>=2.1.2"
    ],
)
