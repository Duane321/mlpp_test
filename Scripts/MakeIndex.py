"""
The goal is produce text like this:


# Machine Learning: A Probabilistic Perspective

Exercise | Solution
------------ | -------------
[MLE for univariate Gaussian](https://duane321.github.io/mlpp_test/TestPDFs/unigaussMLE-ex.pdf)| [Solution](https://duane321.github.io/mlpp_test/TestPDFs/unigaussMLE-sol.pdf)

"""

import pandas as pd 

preamble = """# Machine Learning: A Probabilistic Perspective

"""


git_loc = 'https://duane321.github.io/mlpp_test/TestPDFs/'
private_loc = 'https://duane321.github.io/mlpp_test/TestPDFs/'

name_pairings = pd.read_csv('./TestPDFs/name_pairings.csv')

lines = [preamble]

for i in range(name_pairings.shape[0]):
    full_name = name_pairings['full_name'].iloc[i]
    hashed_name = name_pairings['hashed_name'].iloc[i]
    if name_pairings['level'].iloc[i] == 'public':
        line = '|[' + full_name + '](' + git_loc + hashed_name + '_ex.pdf)|[Solution](' + git_loc + hashed_name + '_sol.pdf)|'
    elif name_pairings['level'].iloc[i] == 'private':
        line = '|[' + full_name + '](' + git_loc + hashed_name + '_ex.pdf)|[Solution](' + private_loc + hashed_name + '_sol.pdf)|'
    else:
        raise ValueError('Incorrect level given for {}'.format(hashed_name))
    lines.append(line+'\n')

with open('index.md','w') as file:
    file.write(''.join(lines))