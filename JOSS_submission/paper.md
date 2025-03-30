---
title: 'pytau: A Python package for streamlined changepoint model analysis in neuroscience'
tags:
  - Python
  - neuroscience
  - changepoint analysis
  - time-series
  - bayesian modeling
authors:
  - name: Abuzar Mahmood
    affiliation: 1
affiliations:
 - name: Independent Researcher
   index: 1
date: 30 March 2025
bibliography: paper.bib
---

# Summary

Analyzing complex biological data, particularly time-series data from neuroscience experiments, often requires sophisticated statistical modeling to identify significant changes in system dynamics. Changepoint models are crucial for detecting abrupt shifts or transitions in neural activity patterns. `pytau` is a Python software package designed to perform streamlined, batched inference for changepoint models across different parameter grids and datasets. It provides tools to efficiently query and analyze the results from sets of fitted models, facilitating the study of dynamic processes in biological systems, such as neural ensemble activity in response to stimuli.

# Statement of need

Understanding how neural populations encode information often involves analyzing activity changes over time, potentially across different experimental conditions, parameters, or subjects. Fitting and comparing complex models like Bayesian changepoint models across numerous datasets or parameter settings can be computationally intensive and logistically challenging. There is a need for tools that streamline this process, enabling researchers to efficiently apply these models in batch, manage the results, and compare outcomes across conditions. `pytau` aims to fill this gap by providing a modularized pipeline specifically for fitting and analyzing changepoint models applied to neuroscience data, enabling efficient comparisons and analysis.

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from collaborators and support during the development of this project.

# References
