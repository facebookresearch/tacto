# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

import os
import re

from setuptools import find_packages, setup

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

install_requires = [line.rstrip() for line in open("requirements/requirements.txt", "r")]

package_data = {
    'tacto': ["config_digit.yml", "config_omnitact.yml"]
}


def read(fname):
    return open(os.path.join(BASE_DIR, fname)).read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="tacto",
    version=find_version("tacto/__init__.py"),
    description="Simulator for vision-based tactile sensors.",
    url="https://github.com/facebookresearch/tacto",
    author="Roberto Calandra",
    author_email="rcalandra@fb.com",
    keywords=["science"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="LICENSE",
    packages=find_packages(),
    package_data=package_data,
    install_requires=install_requires,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.6",
)
