# [FG-AI4NDM Educational Notebooks](https://jeepchinnawat.github.io/edumat-book/)

This collection of tutorial notebooks aims to support machine learning applications and researches for Natural Disaster Management (NDM). The body of work is still in an on-going development so the book will continue to evolve, with the emphasis on open(-source and -data), hazard-related, and geoscientific AI contents.

## Getting started

### Run online

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jeepchinnawat/edumat-book/HEAD)

You can run the whole book with all tutorials online on cloud platform. The button above takes you to Binder with everything in the repository, and it will install required dependencies for you automatically.

Also, every tutorial can be run online individually with Binder or Google Colab, available at the launch button at the top of the respective tutorial page in [the book](https://jeepchinnawat.github.io/edumat-book/).

> **Caution:**
> The caveat is your changes will not be saved unless you download the notebooks before closing the cloud platform.

### Run locally

If you run the notebook locally, it is recommended to do it with a virtual environment or environment management. The steps below use `conda`.

1. Clone this repository
2. Create a new conda (or any other) environment
3. Activate the created environment and install `pip`
4. Run `pip install -r requirements/tutorials.txt`
5. Run `jupyter lab` and try to execute the import cell in `book/penguins.ipynb` to test the dependencies.

### Build the book locally

1. Run `pip install -r requirements/book.txt`
2. Run `jupyter-book build book/`

The markdown and jupyter notebook file in `book/` will then be rendered in HTML pages in `book/_build/html/`.
Open `book/_build/html/index.html` in the browser to see or checked applied changes.