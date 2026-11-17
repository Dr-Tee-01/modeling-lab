"""The four labs: how the models are grouped and ordered in the interface.

A view keeps its technical key (``spread/sir``, ``crossover/dose``...), which is what the API,
the explainers and the tests use; a lab only decides where it is shown and under what name.
"""
from __future__ import annotations

LABS = [
    {"id": "modeling", "name": "Modeling Lab", "short": "Modeling Lab", "icon": "spread", "color": "s1",
     "title": "How things move through a population",
     "views": ["spread/dosing", "spread/bass", "spread/sir", "spread/rumour", "spread/chance", "spread/walk"]},
    {"id": "geometry", "name": "Geometry Lab", "short": "Geometry Lab", "icon": "shape", "color": "s2",
     "title": "Geometry processing from first principles",
     "views": ["shape/fourier", "shape/jaw", "shape/spectral", "shape/curvature", "shape/skinning"]},
    {"id": "timeseries", "name": "Time Series Lab", "short": "Time Series Lab", "icon": "markets", "color": "s3",
     "title": "What can a price series tell us?",
     "views": ["markets/volatility", "markets/returns", "markets/neural",
               "crossover/random-walk", "crossover/dose", "crossover/crash-shapes"]},
    {"id": "rl", "name": "Reinforcement Learning Lab", "short": "RL Lab", "icon": "agents", "color": "s4",
     "title": "Snake: search versus deep Q-learning",
     "views": ["agents/play", "agents/watch", "agents/train", "agents/compare"]},
    {"id": "bigdata", "name": "Big Data Lab", "short": "Big Data Lab", "icon": "bigdata", "color": "c1",
     "title": "How data systems split, move and summarise data",
     "views": ["bigdata/hdfs", "bigdata/mapreduce", "bigdata/mr-joins", "bigdata/rpc", "bigdata/rdd", "bigdata/dag",
               "bigdata/spark-joins", "bigdata/sql", "bigdata/tables", "bigdata/olap", "bigdata/cube", "bigdata/lattice"]},
]
ORDER = {key: (i, j) for i, lab in enumerate(LABS) for j, key in enumerate(lab["views"])}


def lab_of(key: str) -> dict | None:
    return next((lab for lab in LABS if key in lab["views"]), None)
