import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

def density_scatter(x, y, ax = None, sort = True, bins = 20, **kwargs )   :
	"""
	Scatter plot colored by 2d histogram
	"""
	from matplotlib import cm
	from matplotlib.colors import Normalize
	from scipy.interpolate import interpn

	if ax is None :
		fig , ax = plt.subplots(figsize=(4.5,4))
	data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
	z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

	#To be sure to plot all data
	z[np.where(np.isnan(z))] = 0.0

	# Sort the points by density, so that the densest points are plotted last
	if sort :
		idx = z.argsort()
		x, y, z = x[idx], y[idx], z[idx]

	ax.scatter( x, y, c=z, **kwargs )

	norm = Normalize(vmin = np.min(z), vmax = np.max(z))

	return ax

def benchmark(y_true, y_pred, title, savetitle, figdir): 
	y_true = y_true.reshape(-1)
	y_pred = y_pred.reshape(-1)

	mse = np.mean((y_true - y_pred)**2) 
	rmse = np.sqrt(mse) 
	corr, p = spearmanr(y_true.reshape(-1), y_pred.reshape(-1))

	print('RMSE:', rmse)
	print('Corr:', corr)

	if len(y_true)>800:
		ax = density_scatter(y_pred, y_true, bins=[500,500], s=1)
	else:
		plt.figure(figsize=(4,4))
		plt.scatter(y_pred, y_true, s=1)
	mn = max(np.amin(y_pred), np.amin(y_true))
	mx = min(np.amax(y_pred), np.amax(y_true))
	plt.plot(np.unique(y_pred), np.poly1d(np.polyfit(y_pred, y_true, 1))(np.unique(y_pred)), 'y-', linewidth=0.8)
	plt.plot([mn, mx],[mn, mx], 'g-', linewidth=0.8)
	plt.ylabel("True Value")
	plt.xlabel("Predicted Value")
	plt.title(title+'\n RMSE = {:.4f}, Corr = {:.4f}'.format(rmse, corr), fontsize=12)
	if len(y_true)<10000:
		plt.savefig(figdir+'/'+savetitle+'.pdf')
	else:
		plt.savefig(figdir+'/'+savetitle+'.png', dpi=350)

	return rmse, corr
