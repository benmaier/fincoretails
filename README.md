*Note: This is a dummy README for the package you want to build. For a README for the package builder, check out
[PACKAGE_CREATION_README.md](https://github.com/benmaier/fincoretails/blob/main/PACKAGE_CREATION_README.md)*

![logo](https://github.com/benmaier/fincoretails/raw/main/img/Fig1.png)

[![CircleCI](https://circleci.com/gh/benmaier/fincoretails.svg?style=svg)](https://circleci.com/gh/benmaier/fincoretails)

Description of this package goes here.

* repository: https://github.com/benmaier/fincoretails/
* documentation: http://fincoretails.benmaier.org/

```python
from fincoretails.sqrt import get_sqrt_2

print(get_sqrt_2())
```

## Install

    pip install fincoretails

`fincoretails` was developed and tested for 

* Python 3.6
* Python 3.7
* Python 3.8

So far, the package's functionality was tested on Mac OS X and CentOS only.

## Dependencies

`fincoretails` directly depends on the following packages which will be installed by `pip` during the installation process

* `numpy>=1.17`

## Documentation

The full documentation is available at [fincoretails.benmaier.org](http://fincoretails.benmaier.org).

## Changelog

Changes are logged in a [separate file](https://github.com/benmaier/fincoretails/blob/main/CHANGELOG.md).

## License

This project is licensed under the [MIT License](https://github.com/benmaier/fincoretails/blob/main/LICENSE).
Note that this excludes any images/pictures/figures shown here or in the documentation.

## Contributing

If you want to contribute to this project, please make sure to read the [code of conduct](https://github.com/benmaier/fincoretails/blob/main/CODE_OF_CONDUCT.md) and the [contributing guidelines](https://github.com/benmaier/fincoretails/blob/main/CONTRIBUTING.md). In case you're wondering about what to contribute, we're always collecting ideas of what we want to implement next in the [outlook notes](https://github.com/benmaier/fincoretails/blob/main/OUTLOOK.md).

[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v1.4%20adopted-ff69b4.svg)](code-of-conduct.md)

## Dev notes

Fork this repository, clone it, and install it in dev mode.

```bash
git clone git@github.com:YOURUSERNAME/fincoretails.git
make
```

If you want to upload to PyPI, first convert the new `README.md` to `README.rst`

```bash
make readme
```

It will give you warnings about bad `.rst`-syntax. Fix those errors in `README.rst`. Then wrap the whole thing 

```bash
make pypi
```

It will probably give you more warnings about `.rst`-syntax. Fix those until the warnings disappear. Then do

```bash
make upload
```
