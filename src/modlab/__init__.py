"""Modeling Lab: one numerical toolkit, three domains, served live.

Packages
--------
core       shared numerics: RK4, Monte Carlo helpers, PCA, fitting
spread     MC312 models (dosing, Bass, SIR, rumour, decay, random walk)
crossover  market data seen through the spread and shape toolkits
markets    live walk-forward forecasting on top of stocklab
server     the Flask app: JSON API + interactive frontend
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
__version__ = "0.2.0"
