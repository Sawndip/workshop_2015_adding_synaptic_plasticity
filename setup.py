from setuptools import setup

setup(
    name="sPyNNakerExtraModelsPlugin",
    version="2015.002",
    description="Extra models not in PyNN",
    url="https://github.com/SpiNNakerManchester/sPyNNakerExtraModelsPlugin",
    packages=['spynnaker_extra_pynn_models',
              'spynnaker_extra_pynn_models.model_binaries',
              'spynnaker_extra_pynn_models.neural_models',
              'spynnaker_extra_pynn_models.neural_properties',
              'spynnaker_extra_pynn_models.neural_properties.synapse_dynamics',
              'spynnaker_extra_pynn_models.neural_properties.synapse_dynamics.dependences'],
    package_data={'spynnaker_extra_pynn_models.model_binaries': ['*.aplx']},
    install_requires=['SpyNNaker >= 2015.004']
)
