# TAPA-2025-Stylometric-Criticism

Data and code for replicating Bolt et al., "Stylometric Criticism of Latin Literature: From Exploratory Data Analysis to Close Reading," _TAPA_

## Citation

If you use the data or code in this repository, please cite:

> T. J. Bolt, E. D. Adams, Z. Adramerinas, P. J. Burns, T. Dasgupta, A. Deng, E. T. Gianitsos, E. F. Rincon, P. Chaudhuri, and J. P. Dexter. “Stylometric Criticism of Latin Literature: From Exploratory Data Analysis to Close Reading.” *TAPA* 155 (2025): 205–250. https://doi.org/10.1353/apa.2025.a957882.

### BibTeX

```bibtex
@article{bolt2025stylometric,
  author  = {Bolt, Thomas J. and Adams, E. D. and Adramerinas, Z. and Burns, Patrick J. and Dasgupta, T. and Deng, A. and Gianitsos, E. T. and Rincon, E. F. and Chaudhuri, Pramit and Dexter, Joseph P.},
  title   = {Stylometric Criticism of Latin Literature: From Exploratory Data Analysis to Close Reading},
  journal = {TAPA},
  volume  = {155},
  year    = {2025},
  pages   = {205--250},
  doi     = {10.1353/apa.2025.a957882}
}
```
## Repository Structure

The `stylometry` directory contains the Python code for calculating the set of 26 stylometric features. 

To reproduce the full verse and prose datasets used in the article, run `extract_features.py` on the following directories:

* `TAPA_verse_texts_preprocessed`
* `TAPA_prose_texts_preprocessed`

Other parts of the repository are currently under construction. Please check back soon or, for urgent questions, contact Pramit Chaudhuri ([pramit.chaudhuri@austin.utexas.edu](mailto:pramit.chaudhuri@austin.utexas.edu)) or Joseph Dexter ([j.dexter@northeastern.edu](mailto:j.dexter@northeastern.edu)).
