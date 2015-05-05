#coding=utf-8
from essentia import *
from essentia.standard import *
import os
from sklearn.svm import *
from sklearn.metrics import *
from sklearn.cross_validation import *
from sklearn.preprocessing import *
from music21 import *
import sklearn.feature_selection
import numpy as np
import math
from sklearn.linear_model import *
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import codecs

def get_waon_midi():
	filenames = os.listdir('mp3')
	cnt = 0
	for fname in filenames:
		if ".mp3" not in fname:
			continue
		
		cnt += 1
		print cnt
		idx = fname.split('.')[0]
		if os.path.exists('waon_midi/' + idx + '.mid'):
			continue
		#ffmpeg -i mp3/1.mp3 -f wav 1.wav
		cmd = 'ffmpeg -i mp3/' + idx + '.mp3 -f wav ' + idx + '.wav'
		os.system(cmd)
		#/waon -i 2.wav -o mid_2.mid -w 3 -n 4096 -s 2048
		cmd = '/home/gmcather/software/waon -i ' + idx + '.wav -o ' + idx + '.mid -w 3'
		os.system(cmd)
		#cp 1.mid /waon_midi/1.mid
		cmd = 'cp ' + idx + '.mid waon_midi/' + idx + '.mid'
		os.system(cmd) 
		#rm -rf 1.wav 1.mid
		cmd = 'rm -rf ' + idx + '.wav ' + idx + '.mid' 
		os.system(cmd)

def getlines(infile):
	fin = open(infile, "r")
	lines = fin.readlines()
	return lines

def filter_song_list(song_list, drc):
	songs = getlines(song_list)
	_set = set([name.split('.')[0] for name in os.listdir(drc)])
	fout = open(song_list, "w")
	for song in songs:
		if song.strip('\r\n') in _set:
			fout.write(song)
	fout.close()

def save_file(input_list, outfile):
	fout = open(outfile, "w")
	for line in input_list:
		fout.write("\t".join(line) + "\n")
	fout.close()

def extract_audio_feature(filelist, dire):
	fin = open(filelist, "r")
	lines = fin.readlines()
	ext = Extractor(midLevel=False, highLevel=False, rhythm=False, tuning=False, dynamics=False, tonalFrameSize=2048, tonalHopSize=1024)

	cnt = 0
	for line in lines:
		cnt += 1
		print cnt

		fdire, fname = line.strip('\r\n').split('/')
		prefix = fname.split('.')[0]
		
		opath = dire + os.sep + prefix + '.fea'
		if os.path.exists(opath):
			continue
		
		audio = MonoLoader(filename=(fdire + os.sep + fname))()
		#pool = PoolAggregator()(ext(audio))
		pool = PoolAggregator(defaultStats=['mean'])(ext(audio))
		feature_list = pool.descriptorNames()
		feature = []
		for feature_name in feature_list:
			if type(pool[feature_name]) is float:
				feature.append(pool[feature_name])
			else:
				feature.extend(pool[feature_name])
		print 'feature len %d' %len(feature)
		save_file([['\t'.join(str(i) for i in feature)+'\n']], opath)

def extract_midi_feature(filelist, dire):
	lines = getlines(filelist)
	print "yes"
	i = 0
	while i < len(lines):
		line = lines[i].strip('\t\r\n')
		print line
		if "<data_set_id>" in line:
			fname = line.split('/')[-2].split('.')[0] + '.fea'
			i += 1
			data = []
			while i < len(lines):
				line = lines[i].strip('\t\r\n')
				print line
				if "<data_set_id>" in line:
					break
				if "<v>" in line:
					v_string = line[3:len(line)-4]
					data.append(v_string)
				i += 1
			save_file([['\t'.join(data) + '\n']], dire + os.sep + fname)
		else:
			i += 1	

def load_data(song_list, dire):
	X = []
	song_list = getlines(song_list)
	for song in song_list:
		song = song.strip('\r\n')
		opath = dire + os.sep + song + '.fea'
		if os.path.exists(opath):
			line = getlines(opath)[0].strip('\r\n').split('\t')
			X.append([float(i) for i in line])
	return np.array(X)

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

def train_model(X, y):
	#clf = SVR(C=1500000.0, kernel='rbf')
	clf = SVR()
	#clf = LinearSVR()
	#clf = LinearRegression()
	print len(X), len(y)
	print X.shape
	y = y[0:len(X)]
	selector = sklearn.feature_selection.SelectKBest(sklearn.feature_selection.f_regression)
	X = selector.fit_transform(X,y)
	print len(X)
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
	print 'total_score: %lf' %ava_score
	return ava_score

def use_pipeline(X, y):
	y = y[0:len(X)]
	transform = sklearn.feature_selection.SelectPercentile(sklearn.feature_selection.f_regression)
	clf = Pipeline([('anova', transform),('svc', SVR())])
	percentiles = (1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100)
	score_means = list()
	score_stds = list()

	for percentile in percentiles:
		clf.set_params(anova__percentile=percentile)
		this_scores = cross_val_score(clf, X, y, n_jobs=1)
		score_means.append(this_scores.mean())
		score_stds.append(this_scores.std())
		print this_scores

	plt.errorbar(percentiles, score_means, np.array(score_stds))

	plt.title(
   		 'Performance of the SVM-Anova varying the percentile of features selected')
	plt.xlabel('Percentile')
	plt.ylabel('Prediction rate')

	plt.axis('tight')
	plt.show()

def get_half_name(drc, ifile, ofile):
	name_list = []
	for fname in os.listdir(drc):
		name_list.append(fname[:-4])
	name_set = set(name_list)

	fin = codecs.open(ifile, "r", "gbk")
	lines = fin.readlines()
	fin.close()
	for line in lines:
		fid, fname = line.split(u',')[0], line.split(u',')[2]
		#print type(fname.encode('utf8'))
		#print type(fname.decode('utf8'))
		#print type(fname.encode('gbk'))
		#print type(fname.decode('gbk'))
		if fname.encode('utf8') in name_set:
			print fname, type(fname)
			print fid, type(fid)
			try:
				os.rename(drc + os.sep + fname + u'.mid', drc + os.sep + fid + u'.mid')
			except:
				try:
					os.remove(drc + os.sep + fname + u'.mid')
				except:
					pass

	fid_list = [fname[:-4] for fname in os.listdir(drc)]
	fout = open(ofile, "w")
	fid_list.sort()
	for fid in fid_list:
		fout.write(fid + u'.mp3\n')
	fout.close()

def run(song_list, fdrc, ldrc):
	filter_song_list(song_list, fdrc)
	filter_song_list(song_list, ldrc)
	X = load_data(song_list, fdrc)
	X = normalize(X=X, norm='l2', axis=0)

	Y = load_data(song_list, ldrc)
	P, A, D = Y[:,0], Y[:,1], Y[0:,2]
	P_score = train_model(X, P)
	A_score = train_model(X, A)
	D_score = train_model(X, D)
	print 'P_score: %lf A_score: %lf D_score: %lf' %(P_score, A_score, D_score)

if __name__=="__main__":
	#get_waon_midi()
	#extract_audio_feature('mp3.list', 'feature')
	#extract_midi_feature('feature_edit.xml', 'feature4.0')
	run('song2.list', 'feature', 'label')
	run('song2.list', 'feature2.0', 'label')
	run('song2.list', 'feature3.0', 'label')
	run('song2.list', 'feature4.0', 'label')
