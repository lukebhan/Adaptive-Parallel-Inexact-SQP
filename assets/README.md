# README assets

Official wordmarks, downloaded without modification:

- [UC San Diego](https://www.ucsd.edu/_resources/img/logo_UCSD.png)
- [Georgia Tech](https://brand.gatech.edu/sites/default/files/inline-images/GTLogo_RGB.png)
- [UC Berkeley](https://www.berkeley.edu/wp-content/themes/berkeleygateway/img/logo-berkeley.svg)

University marks are excluded from the repository's MIT license.

`burgers-results.png` is Figure 5 rendered from the bundled data. Refresh it with:

```bash
python experiments.py plot --archived
pdftoppm -png -singlefile -scale-to 1800 results/paper/figure_5.pdf assets/burgers-results
```
