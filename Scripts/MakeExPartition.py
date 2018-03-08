"""
This script takes an 'Exercises.tex' file and breaks it up into dictionary
with key-value pairs:

    the key: a string indicating the name of the exercise and whether it's the 
    problem or solution
    
    the value: the body text along with the preample (everything in the 
    Exercises.tex that comes before the first Exercise)

Then saves it as a json file.
"""

import numpy as np
import pandas as pd
import os

#exercise_filename = 'exercises.tex'
exercise_filename = 'exercises-qmr.tex'
output_folder = 'Exercises'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# with open(exercise_filename,'rb') as file:
with open(exercise_filename,'r') as file:
    exercises = file.read()
exercises   = exercises.replace("\r\n", "\n")

ex_sig = '\myexercise{'
sol_sig = '\mysoln{'

def findall(sub, string):
    ''' Gives all the positions of the substring 'sub' in the string 'string'.'''
    i = string.find(sub)
    while i != -1:
        yield i
        i = string.find(sub, i+1)
        
def extract_label(string):
    """Given a string like 'blah blah \label{ex:foo bar} yada yada yada'
    this will return 'foo bar' (the first such match)
    """
    pos = string.index('\\label{ex:')
    string = string[pos+10:]
    return string[:string.index('}')]

def extract_label_myex(string):
    """Given a string like 'blah blah \myexercise{foo bar} yada yada yada'
    this will return 'foobar' (the first such match)
    """
    pos = string.index('\myexercise{')
    string = string[pos+12:]
    return string[:string.index('}')]
    
remove = ['$','\\','^','\n',',','|','%',r'/']
def clean_string(string):
    for rm in remove:
        string = string.replace(rm,'')
    return string
    
ex_positions = list(findall(ex_sig, exercises))
ex_positions.reverse()

pos = exercises.index('\\begin{document}')
preamble = exercises[:pos+16] + '\n'
ending = '{}\n{}\n{}'.format('\\bibliography{/Users/kpmurphy/GDrive/Backup/Latex/bib}', 
                             '\\bibliographystyle{chicago}',
                             '\\end{document}')

partition = {}
#This tells us the hash-name full name pair.
full_names = []
hashed_names = []
level = []
 
while ex_positions:
    ex_pos_i = ex_positions.pop()
    
    if len(ex_positions) == 0:
        ex_next = exercises.index('\\bibliography{')
    else:
        ex_next = ex_positions[-1]
        
    ex_sol_body = exercises[ex_pos_i:ex_next]
    ex_sol_body  = ex_sol_body.replace("\r\n", "\n")
    #Try to find the \label{ex:}. If not found, 
    #we use the natural language exercise title with spaces removed.
    full_name = extract_label_myex(ex_sol_body)
    try:
        ex_name = extract_label(ex_sol_body)
    except:
        ex_name = full_name.replace(' ','')
        print("-------------------------------------")
        print("Missing mylabel. Using '{}'".format(ex_name))
        print()
    ex_name = clean_string(ex_name)
    if full_name[-1] == '*':
        level.append('private')
        full_name = full_name[:-1]
    else:
        level.append('public')
    full_names.append(full_name)
    hashed_names.append(ex_name)
    
    if sol_sig in ex_sol_body:
        sol_pos = ex_sol_body.index(sol_sig)
        partition[ex_name+'_ex'] = preamble + ex_sol_body[:sol_pos] + ending
        partition[ex_name+'_sol'] = preamble + ex_sol_body[sol_pos:] + ending
    else:
        partition[ex_name+'_ex'] = preamble + ex_sol_body + ending

name_pairings = pd.DataFrame({'full_name':full_names,'hashed_name':hashed_names,'level':level})
name_pairings.to_csv(output_folder+'/name_pairings.csv',index=False)

for key, value in partition.items():
    fname = '{}/{}.tex'.format(output_folder, key)
    # fname = '{}.tex'.format(key)
    print('writing {}'.format(fname))
    #with open(fname, 'w') as file:
    # value = DuaneCompSub(value)
    with open(fname, 'w') as file:
    #with open(fname, 'w', newline="\n") as file:
        file.write(value)

# with open(output_folder+'/NamePairings.txt', 'w') as file: 
#     for pair in name_pairings:
#         file.write(str(pair) + '\n')
