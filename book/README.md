# Australian CPI Forecasting

Reproducible CRISP-DM methodology report for the Australian CPI forecasting portfolio project — SARIMA, Elastic Net, Ensemble, SVAR, scenario engine, RBA classifier, and credit stress.

## Building the book

1. Clone this repository.
2. `pip install -r requirements.txt` (ideally in a virtual environment).
3. (Optional) Edit the source files in `australian_cpi_forecasting/`.
4. `jupyter-book clean australian_cpi_forecasting/` to clear existing builds.
5. `jupyter-book build australian_cpi_forecasting/`.

The rendered HTML is built to `australian_cpi_forecasting/_build/html/`.

## Hosting the book

See the [Jupyter Book publishing docs](https://jupyterbook.org/publish/web.html) for deploying via GitHub, GitLab, or Netlify.

## Credits

Built with [Jupyter Book](https://jupyterbook.org/) and the [cookiecutter-jupyter-book](https://github.com/executablebooks/cookiecutter-jupyter-book) template.
