#coding=utf-8
from precess import *
import pickle,pprint

def coef(y_true, y_pred):
	ava_y_true = np.array(([sum(y_true)/len(y_true)])*len(y_true))
	ava_y_pred = np.array(([sum(y_pred)/len(y_pred)])*len(y_pred))
	a = (y_true-ava_y_true).dot(y_pred-ava_y_pred)
	#print type(a)
	b = math.sqrt(sum((y_true-ava_y_true)**2)*sum((y_pred-ava_y_pred)**2))
	#print type(b)
	return a/b

def r2(y_true, y_pred):
	return coef**2

def print_to_file(_list, outfile):
	fout = open(outfile, "a")
	fout.write('  '.join([str(i) for i in _list]) + '\n')
	fout.close()

def feature_select(X, y, k):
	#print X.shape
	selector = sklearn.feature_selection.SelectKBest(sklearn.feature_selection.f_regression, k = k)
	X = selector.fit_transform(X,y)
	feature_selected = selector.get_support(indices=True)
	#print feature_selected
	#pickout = open('data.pkl', 'wb')
	#pickle.dump(feature_selected, pickout, 0)
	#pickout.close()
	print_to_file(feature_selected, 'compare.list')
	#print selector.scores_[feature_selected]
	'''
	index = range(len(selector.scores_))
	index.sort(key=lambda i: -selector.scores_[i])
	print index
	print selector.scores_[index]
	print index[:20]
	print selector.scores_[index[:20]]
	'''
	#print selector.pvalues_[feature_selected]
	#print feature_selected
	#print len(X)
	return X, y, feature_selected

def train_model(X, y, k):
	#clf = SVR(C=1500000.0, kernel='rbf')
	clf = SVR()
	X, y , feature_selected = feature_select(X, y, k)
	kfold = KFold(n=min(len(X),len(y)), n_folds=5, shuffle=False, random_state=None)
	s_score = 0
	for train_index, test_index in kfold:
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		clf = clf.fit(X_train,y_train)
		y_pred = clf.predict(X_test)
		#p_score = r2(y_test, y_pred)
		#p_score = r2_score(y_test, y_pred)
		p_score = coef(y_test, y_pred)
		#print y_test
		#print y_pred
		#print 'per_score: %lf' %p_score
		#print abs(r2_score(y_test, y_pred))**0.5
		s_score += p_score
	ava_score = s_score/len(kfold)
	#print 'total_score: %lf' %ava_score
	return ava_score

def run(song_list, fdrc, ldrc):
	filter_song_list(song_list, fdrc)
	filter_song_list(song_list, ldrc)
	X = load_data(song_list, fdrc)
	X = normalize(X=X, norm='l2', axis=0)
	index = []
	index.extend([3,10,45,74,89])
	index.extend([2, 16, 19, 35])
	#X = X[:,[16]]

	Y = load_data(song_list, ldrc)
	P, A, D = Y[:,0], Y[:,1], Y[0:,2]
	#P_score = train_model(X, P, 1)
	#print 'k=%d Score=%lf\n' %(1, P_score)
	for k in range(1, 400):
		P_score = train_model(X, P, k)
		print 'k=%d Score=%lf\n' %(k, P_score)
	#A_score = train_model(X, A)
	#D_score = train_model(X, D)
	#print 'P_score: %lf\nA_score: %lf\nD_score: %lf\n' %(P_score, A_score, D_score)

def run_merge(song_list, fdrc, fdrc2, ldrc):
	index = []
	index.extend([3,10,45,74,89])
	index.extend([2, 16, 19, 35])
	index.extend(range(404,532))
	filter_song_list(song_list, fdrc)
	filter_song_list(song_list, fdrc2)
	filter_song_list(song_list, ldrc)

	X = load_data(song_list, fdrc)
	X = normalize(X=X, norm='l2', axis=0)
	#X = X[:,index]

	X2 = load_data(song_list, fdrc2)
	X2 = normalize(X=X2, norm='l2', axis=0)
	#X2 = X2[:,index]

	X = np.hstack((X, X2))
	print X.shape

	Y = load_data(song_list, ldrc)
	P, A, D = Y[:,0], Y[:,1], Y[0:,2]
	#P_score = train_model(X, P)
	for k in range(1, 400):
		P_score = train_model(X, P, k)
		print 'k=%d Score=%lf\n' %(k, P_score)

def test_all():
	#run('song2.list', 'feature', 'label')
	run_merge('song2.list', 'feature','feature3.0','label')
	#run('song2.list', 'feature3.0', 'label')
	#run('song2.list', 'feature4.0', 'label')
	#run_merge('song2.list', 'feature2.0', 'feature3.0', 'label')

if __name__=="__main__":
	#get_waon_midi()
	#extract_audio_feature('mp3.list', 'feature')
	#extract_midi_feature('feature_edit.xml', 'feature4.0')
	test_all()
	#get_feature_name('definition.xml', 'feature.nam')
	#init_feature_dict()
	'''
	tdict = init_feature_dict()
	create_arff(tdict, 'song2.list', 'feature2.0', 'label', '515_widi.arff')
	create_arff(tdict, 'song2.list', 'feature3.0', 'label', '515_waon.arff')
	create_arff(tdict, 'song2.list', 'feature4.0', 'label', '515_edit.arff')
	create_arff(tdict, 'song.list', 'feature2.0', 'label', '1999_widi.arff')
	create_arff(tdict, 'song.list', 'feature3.0', 'label', '1999_waon.arff')
	'''
