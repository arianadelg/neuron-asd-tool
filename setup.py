from setuptools import setup, find_packages
setup(
    name="neuron-asd",
    version="1.1.0",
    description="Open platform for exploring receptor agonist/inhibitor modulations toward a typically-developing EEG profile in autism",
    author="Alvarado, Y. J.; Cardozo-Urdaneta, A.; Lossada, C.; Quintero, M.; Delgado, L.; Gonzalez-Paz, L.",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "mne>=1.5","fooof>=1.0","numpy>=1.23","scipy>=1.9",
        "pandas>=1.5","matplotlib>=3.6","scikit-learn>=1.2","tensorflow-cpu>=2.12",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
