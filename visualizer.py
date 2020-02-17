import pandas as pd
import numpy as np
import plotly.graph_objs as go
import chart_studio . plotly as py
from plotly.graph_objs import *

def scatter_with_color_dimension_graph (features,target,layout_labels):
	''' Scatter with color dimension graph to visualize the density of the
	Given feature with target
	param feature :
	param target :
	param layout_labels :
	retrun :
	'''
	trace1 = go.Scatter( y = features,mode ='markers ',marker = dict (size = 6,color = target,colorscale = 'Viridis ', showscale = True))

	layout = go.Layout(title = layout_labels [2] ,
	xaxis = dict(title = layout_labels [0]) , yaxis = dict ( title = layout_labels [1]))
	data = [trace1]
	fig = Figure( data =data , layout = layout )
	print(" images ")
	fig.show ()

def main ():
	DATASET_PATH = './ glass_data_labeled . csv '
	df = pd. read_csv ( DATASET_PATH )

	features = df['Mg ']
	targets = df['Type ']

	xlabel = 'Data Index '
	ylabel = 'Mg Value '
	graph_title = 'Mg -- Glass Type Density Graph '
	graph_labels = [ xlabel , ylabel , graph_title ]

	scatter_with_color_dimension_graph(features,targets,graph_labels)

 if __name__ == " __main__ ":
	 main ()