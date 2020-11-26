import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="iidfile",
    version="0.2.0",
    author="aeaeaeae",
    author_email="ouououou@aeaeaeae.io",
    description="File parser for .iid files, for storing labeled image segmentation data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aeaeaeaeaeae/iidformat",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
    ],
    python_requires='>=3.6',
)