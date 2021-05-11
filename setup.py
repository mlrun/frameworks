from typing import List
import os
import logging
import itertools

try:
    from setuptools import setup
    from setuptools.command.install import install
except ImportError:
    from distutils.core import setup
    from distutils.command.install import install

logger = logging.getLogger()


class FrameworkKeys:
    BASE = "Base"
    PYTORCH = "PyTorch"
    # LIGHTNING = "Lightning"
    # TENSORFLOW = "Tensorflow"
    KERAS = "Keras"


FRAMEWORKS_REQUIREMENTS = {
    FrameworkKeys.BASE: "requirements.txt",
    FrameworkKeys.PYTORCH: os.path.join("frameworks/pytorch", "requirements.txt"),
    # FrameworkKeys.LIGHTNING: os.path.join("frameworks/lightning", "requirements.txt"),
    # FrameworkKeys.TENSORFLOW: os.path.join("frameworks/tensorflow", "requirements.txt"),
    FrameworkKeys.KERAS: os.path.join(
        "frameworks/keras", "requirements.txt"
    ),
}

FRAMEWORKS_PACKAGES = {
    FrameworkKeys.BASE: ["frameworks", "frameworks.base"],
    FrameworkKeys.PYTORCH: ["frameworks.pytorch", "frameworks.pytorch.callbacks"],
    # FrameworkKeys.LIGHTNING: ["frameworks.lightning"],
    # FrameworkKeys.TENSORFLOW: ["frameworks.tensorflow"],
    FrameworkKeys.KERAS: [
        "frameworks.keras",
        "frameworks.keras.callbacks",
    ],
}


class InstallCommand(install):
    """
    Custom 'install' command to include frameworks attributes in the parameter 'install-option'.
    For example, installing mlrun/frameworks for pytorch will be as follows:
    pip install mlrun-frameworks --install-option="--pytorch"
    """

    user_options = install.user_options + [
        ("pytorch", None, "install only the mlrun framework extention for pytorch"),
        # ('lightning', None, "install only the mlrun framework extention for pytorch-lightning"),
        # ('tensorflow', None, "install only the mlrun framework extention for tensorflow"),
        ('keras', None, "install only the mlrun framework extention for tensorflow.keras")
    ]

    def initialize_options(self):
        """
        Initialize the new frameworks attributes with 'None'.
        """
        install.initialize_options(self)
        self.pytorch = None
        # self.lightning = None
        # self.tensorflow = None
        self.keras = None

    def finalize_options(self):
        """
        Parse the user's preference. If no framework was given, all frameworks will be installed. Each attribute will be
        'None' if it was not given and 1 if it was given.
        """
        install.finalize_options(self)
        if (
            self.pytorch
            # or self.lightning
            # or self.tensorflow
            or self.keras
        ):
            if not self.pytorch:
                FRAMEWORKS_REQUIREMENTS.pop(FrameworkKeys.PYTORCH)
                FRAMEWORKS_PACKAGES.pop(FrameworkKeys.PYTORCH)
            # if not self.lightning:
            #     FRAMEWORKS_REQUIREMENTS.pop(FrameworkKeys.LIGHTNING)
            #     FRAMEWORKS_PACKAGES.pop(FrameworkKeys.LIGHTNING)
            # if not self.tensorflow:
            #     FRAMEWORKS_REQUIREMENTS.pop(FrameworkKeys.TENSORFLOW)
            #     FRAMEWORKS_PACKAGES.pop(FrameworkKeys.TENSORFLOW)
            if not self.keras:
                FRAMEWORKS_REQUIREMENTS.pop(FrameworkKeys.KERAS)
                FRAMEWORKS_PACKAGES.pop(FrameworkKeys.KERAS)

    def run(self):
        """
        Run the 'install' command, notifying the user on his selected frameworks.
        """
        logger.info(
            "Installing the mlrun extensions for the following frameworks: {}".format(
                list(FRAMEWORKS_PACKAGES.keys())
            )
        )
        install.run(self)


def get_requirements() -> List[str]:
    """
    Get the full requirements list from the user's frameworks input and their 'requirements.txt' files.
    :return: List of all the packages requirements.
    """
    # Initialize the requirements list:
    requirements = []

    # Collect all requirements.txt files needed:
    requirements_txt_files = list(FRAMEWORKS_REQUIREMENTS.values())

    # Add all the requirements from each of the requirements.txt files:
    for requirement_txt_file in requirements_txt_files:
        with open(requirement_txt_file) as requirements_file:
            requirements += [
                requirement.strip()
                for requirement in requirements_file
                if (
                    requirement not in requirements
                    and (requirement[0] != "#" and requirement.strip() != "")
                )
            ]

    return requirements


# Supports the following pattern: pip install mlrun-frameworks --install-option="--<FRAMEWORK_NAME>"
setup(
    cmdclass={"install": InstallCommand},
    name="mlrun-frameworks",
    version="0.1.0",
    description="MLRun extension packages for deep learning frameworks. Currently supporting PyTorch.",
    author="Yaron Haviv",
    author_email="yaronh@iguazio.com",
    license="MIT",
    url="https://github.com/mlrun/frameworks",
    packages=list(itertools.chain.from_iterable(FRAMEWORKS_PACKAGES.values())),
    install_requires=get_requirements(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries",
    ],
    zip_safe=False,
)
