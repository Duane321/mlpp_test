"""
The goal is produce text like this:


# Machine Learning: A Probabilistic Perspective

Exercise | Solution
------------ | -------------
[MLE for univariate Gaussian](https://duane321.github.io/mlpp_test/TestPDFs/unigaussMLE-ex.pdf)| [Solution](https://duane321.github.io/mlpp_test/TestPDFs/unigaussMLE-sol.pdf)

"""

import pandas as pd
import sys, os

lines = ['# Machine Learning: A Probabilistic Perspective\n',
 '\n',
 '| Exercise  | Solution |\n',
 '| ------------- | ------------- |\n']


git_loc = 'https://duane321.github.io/mlpp_test/PublicPDFs/'
private_loc = 'https://duane321.github.io/mlpp_test/PublicPDFs/'

# sys.path.insert(1, os.path.join(sys.path[0], '../PublicPDFs'))
# for pt in sys.path:
#     print(pt)
name_pairings = pd.read_csv('../PublicPDFs/name_pairings.csv')

for i in range(name_pairings.shape[0]):
    full_name = name_pairings['full_name'].iloc[i]
    hashed_name = name_pairings['hashed_name'].iloc[i]
    if name_pairings['level'].iloc[i] == 'public':
        line = '| [' + full_name + '](' + git_loc + hashed_name + '_ex.pdf) | [Solution](' + git_loc + hashed_name + '_sol.pdf) |\n'
    elif name_pairings['level'].iloc[i] == 'private':
        line = '| [' + full_name + '](' + git_loc + hashed_name + '_ex.pdf) | [Solution](' + private_loc + hashed_name + '_sol.pdf) |\n'
    else:
        raise ValueError('Incorrect level given for {}'.format(hashed_name))
    lines.append(line)

# lines = ['# Machine Learning: A Probabilistic Perspective\n',
#  '\n',
#  '| Exercise  | Solution |\n',
#  '| ------------- | ------------- |\n',
#  '| [Subderivative of the hinge loss function](https://duane321.github.io/mlpp_test/TestPDFs/subgradHinge_ex.pdf)  | [Solution](https://duane321.github.io/mlpp_test/TestPDFs/subgradHinge_sol.pdf)  |\n',
#  '| [Reproducing kernel property](https://duane321.github.io/mlpp_test/TestPDFs/reproducing_ex.pdf)  | [Solution](https://duane321.github.io/mlpp_test/TestPDFs/reproducing_sol.pdf)  |\n',
#  '| [Orthogonal matrices](https://duane321.github.io/mlpp_test/TestPDFs/orthogonalMatrices_ex.pdf)  | [Solution](https://duane321.github.io/mlpp_test/TestPDFs/orthogonalMatrices_sol.pdf)  |\n']
with open('../index.md','w') as file:
    file.write(''.join(lines))