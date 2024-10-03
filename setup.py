from setuptools import setup, find_packages

import os
lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = f"{lib_folder}/requirements.txt"
install_requires = [] # Here we'll add: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

print(install_requires)

setup(
    name="synthesize",
    version='1.0',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'synthesize.Case': ['BRCASubtypeSel_test.csv'],
        'synthesize.Case': ['BRCASubtypeSel_train_epoch285_CVAE1-20_generated.csv'],
        'synthesize': ['metric_generated_30.csv'],
        'synthesize': ['metric_real_30.csv'],
    },
    install_requires = install_requires,
    python_requires='>=3.6',
)