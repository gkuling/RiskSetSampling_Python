import pandas as pd

from RiskSetSampling import RiskSetSampling

# load data that was printed to a csv from the Epi package in R
diet = pd.read_csv('./diet.csv')

print(diet.head(10))

# run risk set sampling
dietcc = RiskSetSampling(data=diet,
                         entry='doe',
                         exit='dox',
                         fail='fail',
                         origin='dob',
                         controls=2,
                         include=['energy'],
                         match='job',
                         silent=False)

print(dietcc.head(10))
