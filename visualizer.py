import pandas as pd
import numpy as np
import plotly.graph_objs as go
import chart_studio.plotly as py
import sys
import matplotlib.pyplot as plt
from pandas import DataFrame
from plotly.graph_objs import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def scatter_with_color_dimension_graph (features,target,layout_labels):
	''' Scatter with color dimension graph to visualize the density of the
	Given feature with target
	param feature :
	param target :
	param layout_labels :
	retrun :
	'''
	trace1 = go.Scatter(y = features, mode ='markers', marker = dict(size = 6, color = target, colorscale = 'Viridis', showscale = True))
	layout = go.Layout(title = layout_labels[2], xaxis = dict(title = layout_labels[0]), yaxis = dict(title = layout_labels[1]))
	data = [trace1]
	fig = Figure(data = data, layout = layout)
	print("images")
	fig.show()

def correlationMatrix(df,targets):
	file = open("corrMatrix.txt", "w+")

	typeIndexEnd = [0] * 7
	index = 0
	for i in range(1,8):
		while index <= 213:
			if targets[index] == i:
				typeIndexEnd[i - 1] = index
			index += 1
		index = 0

	for i in typeIndexEnd:
		if i == 0:
			typeIndexEnd.remove(i)

	typeIndexEnd.insert(0,0)

	for i in range(0,len(typeIndexEnd) - 1):
		start = typeIndexEnd[i]
		end = typeIndexEnd[i+1]
		dataframe = [df['RI'][start:end],df['Na'][start:end],df['K'][start:end],df['Mg'][start:end],df['Al'][start:end],df['Ca'][start:end],df['Si'][start:end],df['Ba'][start:end],df['Fe'][start:end]]
		test = pd.DataFrame(data=df.iloc[typeIndexEnd[i]:typeIndexEnd[i+1]], columns=['RI','Na','K','Mg','Al','Ca','Si','Ba','Fe'])
		# print(str(test.corr()))
		file.write(str(test.corr()))
		file.write('\n')

def elbowPoint(dataframe):
	kval = range(1, 10)
	inertias = [] #minimize inertia
	for k in kval:
	    # Create a KMeans instance with k clusters: model
	    model = KMeans(n_clusters=k)
	    
	    # Fit model to samples
	    model.fit(dataframe.iloc[:,:3])
	    
	    # Append the inertia to the list of inertias
	    inertias.append(model.inertia_)
	    
	plt.plot(kval, inertias, '-o', color='black')
	plt.xlabel('number of clusters, k')
	plt.ylabel('inertia')
	plt.xticks(kval)
	plt.show()

def explainedVariance(pca):
	#used to identify min. features for pca
	plt.figure()
	plt.plot(np.cumsum(pca.explained_variance_ratio_))
	plt.xlabel('Component Count')
	plt.ylabel('Variance')
	plt.title('Glass Dataset Explained Variance')
	plt.show()	

	features = range(pca.n_components_)
	plt.bar(features, pca.explained_variance_ratio_, color='black')
	plt.xlabel('PCA features')
	plt.ylabel('variance %')
	plt.xticks(features)
	plt.show()

def main ():
	DATASET_PATH = 'glass_data_labeled.csv'
	df = pd.read_csv(DATASET_PATH)
	listOfFeatures = ['RI','Na','K','Mg','Al','Ca','Si','Ba','Fe']
	features = [df['RI'],df['Na'],df['K'],df['Mg'],df['Al'],df['Ca'],df['Si'],df['Ba'],df['Fe']]
	targets = df['Type']
	xlabel = 'Data Index'
	ylabel = 'Mg Value'
	graph_title = 'Mg -- Glass Type Density Graph'
	graph_labels = [xlabel, ylabel, graph_title]

	pd.set_option('display.max_rows',None)
	pd.set_option('display.max_columns',None)
	pd.set_option('display.width',None)
	pd.set_option('display.max_colwidth',-1)
	np.set_printoptions(threshold=sys.maxsize)

	# scatter_with_color_dimension_graph(features[3],targets,graph_labels)

	features = df.loc[:,listOfFeatures].values
	standardFeatures = StandardScaler().fit_transform(features)
	correlationMatrix(df,targets)

	#PCA
	pca = PCA(n_components = 6) #value chosen according to explained variance
	components = pca.fit_transform(standardFeatures)
	componentDF = pd.DataFrame(data=components)

	# explainedVariance(pca)

	# plt.scatter(componentDF[0],componentDF[1], alpha=.5, color='black')
	# plt.xlabel('Component 1')
	# plt.ylabel('Component 2')
	# plt.show()

	elbowPoint(componentDF)
	kMeansModel = KMeans(n_clusters = 4)
	kMeansModel.fit(componentDF)
	prediction = kMeansModel.predict(componentDF)

	plt.scatter(componentDF[0], componentDF[1], c=prediction, s=100, cmap='viridis')
	centers = kMeansModel.cluster_centers_
	plt.scatter(centers[:,0], centers[:,1], c='black', s=200, alpha=0.5)
	plt.show()

if __name__ == "__main__":
	main()