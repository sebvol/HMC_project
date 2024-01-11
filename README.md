# [Hidden Markov Models project](https://github.com/SinPalo/Hidden_Markov_Models_project)

![Python](https://img.shields.io/badge/Python-3.10.12-blue.svg)



In this project, we implement the computational techniques presented in the article [1]. Our goal is to simulate mid-price estimations which is often hindered by limited data transparency. By applying particle filtering and sequential Monte-Carlo methods as outlined in the article, we aim to capture the nuances of bond data and overcome the shortcomings of traditional estimation filters. This simulation project seeks to provide a practical application of the theoretical framework proposed by Olivier Guéant and Jiang Pu, offering insights into the market's pricing mechanisms and enhancing decision-making tools for market analysts.

*For a detailed explanation of each implementation please refer to the comments in the file.*



## Getting started 🚀

**We provide necessary informations to run our code and do the simulation.**

### Particle Filter

To run the simulation, make sure you have a data file named `BMOQTreasuryQuotes.xlsx` with the appropriate structure as the script will attempt to load this file. The main execution of the script runs a simulation for 1000 time steps and plots the results to `fig.png`.

```bash
python Particle_Filter_project.py
```

### PMCMC

```bash
python pmcmc.py
```
Sadly due to the difficulties with the data, we did manage to make the pmcmc work in time.


## Authors 🧑‍💻

- Alban Derepas
- Haocheng Liu
- Sinuhé Hinojosa
- Sébastien Vol



## Reference

[1] Guéant, O., & Pu, J. (2018). Mid-price estimation for European corporate bonds: a particle filtering approach. *Market microstructure and liquidity*, *4*(01n02), 1950005.
