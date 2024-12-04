# Installation

## Run online

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jeepchinnawat/edumat-book/HEAD)

You can run the whole book with all tutorials online on cloud platform. The button above (also in the first page) takes you to Binder with everything in the [repository](https://github.com/jeepchinnawat/edumat-book), and it will install required dependencies for you automatically.

Also, every tutorial can be run online individually with Binder or Google Colab, available at the launch button at the top of the respective page.

```{caution}
The caveat is your changes will not be saved unless you download the notebooks before closing the cloud platform.
```

## Run locally

If you run the notebook locally, it is recommended to do it with a virtual environment or environment management. The steps below use `conda`.

1. Clone this repository
2. Create a new conda (or any other) environment
3. Activate the created environment and install `pip`
4. Run `pip install -r requirements/tutorials.txt`
5. Run `jupyter lab` and try to execute the import cell in `book/penguins.ipynb` to test the dependencies.
