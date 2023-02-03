import os
import numpy as np
import pandas as pd
import csv


# helper file to automate DVP storage as CSV
def write_eval(csv_file, eval, name = 'base'):

	# get the expected test columns
	columns = ['Evaluation'] + [x for x in eval.keys()]

	# read in test report if it exists
	if os.path.isfile(csv_file): 
		data_ledger = pd.read_csv(csv_file, header=0, index_col = None)

		# verify the columns match current standards 
		# if not, make a new ledger
		if len(columns) != len(data_ledger.columns) or not (columns == data_ledger.columns).all():
			data_ledger = pd.DataFrame([], columns = columns)

    # otherwise generate a fresh report
	else:
		data_ledger = pd.DataFrame([], columns = columns)
       	
	# generate the proper dataframe
	output = {'Evaluation' : name}
	output = {**output, **eval}
	data_ledger = data_ledger.append(output, ignore_index = True)

	# write to disc
	data_ledger.to_csv(csv_file, index=False)