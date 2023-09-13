import setuptools

setuptools.setup(
    name="film",
    version="0.1",
    description="FILM Frame Interpolation",
    long_description="Pytorch implementation of FILM: Frame Interpolation for Large Motion",
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    include_package_data=True,
    python_requires='>=3.7',
)
