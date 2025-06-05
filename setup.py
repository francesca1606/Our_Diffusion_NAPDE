from setuptools import setup, find_packages

setup(
    name='our_diffusion_napde',
    version='0.1',
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),
    entry_points={
        'console_scripts': [
            'run-train=training.train:main',
            'run-diffusion=diffusion.generate_diffusion:main',
        ]
    },
    include_package_data=True,
)
