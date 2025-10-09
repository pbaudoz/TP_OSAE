from setuptools import setup, find_packages

setup(name="tposae",
      version="0.1.0",
      author="Pierre Baudoz and the THD2 team",
      description="Control code (and simulator) for optical projects.",
      url="https://github.com/pbaudoz/TP_OSAE",
      license="TBD",
      packages=find_packages(),
      package_data={'tposae': ['user_interface/assets/*']},
      classifiers=[
                        "Programming Language :: Python :: 3"
                  ],
      python_requires='>=3.6',
      zip_safe=False,
      install_requires=[],
      entry_points={
            "console_scripts": [
                  "tposae=tposae.cli_interface:main"
            ],
      },
      )
