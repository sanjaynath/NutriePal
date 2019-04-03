import flask
from flask import Flask, request, render_template
from sklearn.externals import joblib
import numpy as np
from scipy import misc
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from keras.models import load_model
from werkzeug.utils import secure_filename
import os
import pandas as pd
from flask import jsonify
import pickle
import re
import nltk
import heapq 
from operator import itemgetter
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')

@app.route('/history', methods=['POST'])
def history_prediction():
	if request.method=='POST':
		x='sanjay'

		
		user_vec = np.zeros((1,1735))
		user_personal_recommended = []
		with open('pkl/history.pkl', 'rb') as f:
			user_hist = pickle.load(f)
		with open('pkl/pkl/pkl/corpora.pkl', 'rb') as f:
			corpora = pickle.load(f)
		with open('pkl/pkl/food_idx.pkl', 'rb') as f:
			food_idx = pickle.load(f)
		with open('pkl/idx_food.pkl', 'rb') as f:
			idx_food = pickle.load(f)

		tfidf = TfidfVectorizer()
		X = tfidf.fit_transform(corpora).toarray()

		user_hist.reverse()
		count=0
		for i in user_hist:
			count+=1
			if count<4:
				temp = tfidf.transform([corpora[food_idx[i]]])
				user_vec+=temp

		user_vec = user_vec/3
		user_recommended_names = []
		p = cosine_similarity(X,user_vec)
		o = heapq.nlargest(8,range(len(p)),p.take)
		user_personal_recommended.append(o)
		for i in range(3,len(user_personal_recommended[0])):
			user_recommended_names.append(idx_food[user_personal_recommended[0][i]])
		print(user_recommended_names)

		#user_hist.reverse()
		user_history = []
		print(user_hist)
		for i in range(4):
			user_history.append(user_hist[i])

		return flask.render_template('history.html',item1=user_recommended_names[0],item2=user_recommended_names[1],item3=user_recommended_names[2],item4=user_recommended_names[3],item5=user_recommended_names[4],
			h1=user_history[0],h2=user_history[1],h3=user_history[2])




UPLOAD_FOLDER = 'static/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
@app.route('/predict', methods=['POST'])
def make_prediction():
	if request.method=='POST':

		# get uploaded image file if it exists
		file = request.files['image']
		if not file: return render_template('index.html', label="No file")
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'],'example.jpg'))

		food_dict = {'baby_back_ribs': 0,
					 'chicken_wings': 1,
					 'chocolate_cake': 2,
					 'club_sandwich': 3,
					 'dumplings': 4,
					 'edamame': 5,
					 'french_fries': 6,
					 'fried_rice': 7,
					 'hamburger': 8,
					 'ice_cream': 9,
					 'macaroni_and_cheese': 10,
					 'macarons': 11,
					 'miso_soup': 12,
					 'mussels': 13,
					 'pizza': 14,
					 'prime_rib': 15,
					 'samosa': 16,
					 'spaghetti_bolognese': 17,
					 'spring_rolls': 18,
					 'waffles': 19}

		#preprocess image for resnet
		img = image.load_img(file, target_size=(224, 224))
		#x = Image.open(file)
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)

		#predict label
		pred = model.predict(x,verbose = 1)
		predicted_class_indices = np.argmax(pred, axis = 1)
		top_preds_indices = pred[0].argsort()[-3:][::-1] # get top 5 predictions
		top_preds_labels = []
		for j in top_preds_indices:
			for key, value in food_dict.items():
				if j == value:
					top_preds_labels.append(key)
		label = top_preds_labels[0]

		with open('pkl/history.pkl', 'rb') as f:
			user_hist = pickle.load(f)
		user_hist.append(label)
		with open('pkl/pkl/history.pkl', 'wb') as f:
			pickle.dump(user_hist,f)



		'''
			**********. nutrient data. ***************

		'''
		df = pd.read_csv('nutrition_values2.csv')
		dic = df.loc[df['product_name'] == label].to_dict()
		for key, val in dic.items():
			try:
				dic[key] = val[next(iter(val))]
			except StopIteration as e:
				print(e)
				break
		eg = dic['energy_100g']
		cb = dic['carbohydrates_100g']
		pr = dic['proteins_100g']
		ft = dic['fat_100g']
		fb = dic['fiber_100g']
		cl = dic['cholesterol_100g']
		dic2 = dic
		dic_json = jsonify(dic)
		print(dic)


		'''
			**********. recommend based on recipe. ***************

		'''
		with open('pkl/food_indices.pkl', 'rb') as f:
			food = pickle.load(f)	
		with open('pkl/recipe_recommendations_names.pkl', 'rb') as f:
			rname = pickle.load(f)
		recipe_recomm = rname[food[label]]
		recipe_recomm_json = jsonify(recipe_recomm)
		print(recipe_recomm)

		dic = df.loc[df['product_name'] == recipe_recomm[1]].to_dict()
		for key, val in dic.items():
			try:
				dic[key] = val[next(iter(val))]
			except StopIteration as e:
				print(e)
				break

		rrec1 = recipe_recomm[1]
		rrecimg1 = '../static/' + recipe_recomm[1] + '.jpg'
		print(dic)
		egr1 = dic['energy_100g']
		cbr1 = dic['carbohydrates_100g']
		prr1 = dic['proteins_100g']
		ftr1 = dic['fat_100g']
		fbr1 = dic['fiber_100g']
		clr1 = dic['cholesterol_100g']

		dic = df.loc[df['product_name'] == recipe_recomm[2]].to_dict()
		for key, val in dic.items():
			try:
				dic[key] = val[next(iter(val))]
			except StopIteration as e:
				print(e)
				break
		rrec2 = recipe_recomm[2]
		rrecimg2 = '../static/' + recipe_recomm[2] + '.jpg'
		print(dic)
		egr2 = dic['energy_100g']
		cbr2 = dic['carbohydrates_100g']
		prr2 = dic['proteins_100g']
		ftr2 = dic['fat_100g']
		fbr2 = dic['fiber_100g']
		clr2 = dic['cholesterol_100g']


		dic = df.loc[df['product_name'] == recipe_recomm[3]].to_dict()
		for key, val in dic.items():
			try:
				dic[key] = val[next(iter(val))]
			except StopIteration as e:
				print(e)
				break
		rrec3 = recipe_recomm[3]
		rrecimg3 = '../static/' + recipe_recomm[3] + '.jpg'
		print(dic)
		egr3 = dic['energy_100g']
		cbr3 = dic['carbohydrates_100g']
		prr3 = dic['proteins_100g']
		ftr3 = dic['fat_100g']
		fbr3 = dic['fiber_100g']
		clr3 = dic['cholesterol_100g']







		'''
			**********. recommend based on nutrients. ***************

		'''
		knn_model = joblib.load('pkl/knn.pkl')
		nutrition_df = pd.read_csv('nutrition_values2.csv', header=0)
		nutrition_values_df = nutrition_df.drop(["product_name", "sugars_100g","Unnamed: 0"], axis=1)
		dic = nutrition_values_df[nutrition_df['product_name'] == label]
		d = dic.values.tolist()
		print("nutrient values--------------")
		print(d)
		p=[]
		p.append(d)
		
		try:
			distances, indices = knn_model.kneighbors(d, n_neighbors=5)
			#nut_recomm = [nutrition_df.loc[i]['product_name'] for i in indices[0]]
			#dic = df.loc[df['product_name'] == nut_recomm[1]].to_dict()
		except:
			return render_template('index.html', label=label,
			nut_recomm='',recipe_recomm=recipe_recomm,
			eg=eg,cb=cb,pr=pr,ft=ft,fb=fb,cl=cl,
			nrecimg1='',nrec1='',egn3 ='',cbn3 ='',prn3 ='',ftn3='',fbn3='',cln3='',
			nrecimg2='',nrec2='',egn2 ='',cbn2 ='',prn2 ='',ftn2='',fbn2='',cln2='',
			nrecimg3='',nrec3='',egn1 ='',cbn1 ='',prn1 ='',ftn1='',fbn1='',cln1='',
			rrecimg1=rrecimg1,rrec1=rrec1,egr3 =egr3,cbr3 =cbr3,prr3 =prr3,ftr3=ftr3,fbr3=fbr3,clr3=clr3,
			rrecimg2=rrecimg2,rrec2=rrec2,egr2 =egr2,cbr2 =cbr2,prr2 =prr2,ftr2=ftr2,fbr2=fbr2,clr2=clr2,
			rrecimg3=rrecimg3,rrec3=rrec3,egr1=egr1, cbr1 =cbr1,prr1 =prr1,ftr1=ftr1,fbr1=fbr1,clr1=clr1)
			#break
		
		
		distances, indices = knn_model.kneighbors(d, n_neighbors=5)
		nut_recomm = [nutrition_df.loc[i]['product_name'] for i in indices[0]]
		nut_recomm_json = jsonify(nut_recomm)
		print(nut_recomm)

		dic = df.loc[df['product_name'] == nut_recomm[1]].to_dict()
		for key, val in dic.items():
			try:
				dic[key] = val[next(iter(val))]
			except StopIteration as e:
				print(e)
				break
		nrec1 = nut_recomm[1]
		nrecimg1 = '../static/' + nut_recomm[1] + '.jpg'
		egn1 = dic['energy_100g']
		cbn1 = dic['carbohydrates_100g']
		prn1 = dic['proteins_100g']
		ftn1 = dic['fat_100g']
		fbn1 = dic['fiber_100g']
		cln1 = dic['cholesterol_100g']

		dic = df.loc[df['product_name'] == nut_recomm[2]].to_dict()
		for key, val in dic.items():
			try:
				dic[key] = val[next(iter(val))]
			except StopIteration as e:
				print(e)
				break
		nrec2 = nut_recomm[2]
		nrecimg2 = '../static/' + nut_recomm[2] + '.jpg'
		egn2 = dic['energy_100g']
		cbn2 = dic['carbohydrates_100g']
		prn2 = dic['proteins_100g']
		ftn2 = dic['fat_100g']
		fbn2 = dic['fiber_100g']
		cln2 = dic['cholesterol_100g']

		dic = df.loc[df['product_name'] == nut_recomm[3]].to_dict()
		for key, val in dic.items():
			try:
				dic[key] = val[next(iter(val))]
			except StopIteration as e:
				print(e)
				break
		nrec3 = nut_recomm[3]
		nrecimg3 = '../static/' + nut_recomm[3] + '.jpg'
		egn3 = dic['energy_100g']
		cbn3 = dic['carbohydrates_100g']
		prn3 = dic['proteins_100g']
		ftn3 = dic['fat_100g']
		fbn3 = dic['fiber_100g']
		cln3 = dic['cholesterol_100g']



		#recommend based on recipe
		with open('pkl/food_indices.pkl', 'rb') as f:
			food = pickle.load(f)	
		with open('pkl/recipe_recommendations_names.pkl', 'rb') as f:
			rname = pickle.load(f)
		recipe_recomm = rname[food[label]]
		recipe_recomm_json = jsonify(recipe_recomm)
		print(recipe_recomm)

		dic = df.loc[df['product_name'] == recipe_recomm[1]].to_dict()
		for key, val in dic.items():
			try:
				dic[key] = val[next(iter(val))]
			except StopIteration as e:
				print(e)
				break

		rrec1 = recipe_recomm[1]
		rrecimg1 = '../static/' + recipe_recomm[1] + '.jpg'
		print(dic)
		egr1 = dic['energy_100g']
		cbr1 = dic['carbohydrates_100g']
		prr1 = dic['proteins_100g']
		ftr1 = dic['fat_100g']
		fbr1 = dic['fiber_100g']
		clr1 = dic['cholesterol_100g']

		dic = df.loc[df['product_name'] == recipe_recomm[2]].to_dict()
		for key, val in dic.items():
			try:
				dic[key] = val[next(iter(val))]
			except StopIteration as e:
				print(e)
				break
		rrec2 = recipe_recomm[2]
		rrecimg2 = '../static/' + recipe_recomm[2] + '.jpg'
		print(dic)
		egr2 = dic['energy_100g']
		cbr2 = dic['carbohydrates_100g']
		prr2 = dic['proteins_100g']
		ftr2 = dic['fat_100g']
		fbr2 = dic['fiber_100g']
		clr2 = dic['cholesterol_100g']


		dic = df.loc[df['product_name'] == recipe_recomm[3]].to_dict()
		for key, val in dic.items():
			try:
				dic[key] = val[next(iter(val))]
			except StopIteration as e:
				print(e)
				break
		rrec3 = recipe_recomm[3]
		rrecimg3 = '../static/' + recipe_recomm[3] + '.jpg'
		print(dic)
		egr3 = dic['energy_100g']
		cbr3 = dic['carbohydrates_100g']
		prr3 = dic['proteins_100g']
		ftr3 = dic['fat_100g']
		fbr3 = dic['fiber_100g']
		clr3 = dic['cholesterol_100g']


		#picture_path='../static/apple_pie.jpg'
		return render_template('index.html', label=label,
			nut_recomm=nut_recomm,recipe_recomm=recipe_recomm,
			eg=eg,cb=cb,pr=pr,ft=ft,fb=fb,cl=cl,
			nrecimg1=nrecimg1,nrec1=nrec1,egn3 =egn3,cbn3 =cbn3,prn3 =prn3,ftn3=ftn3,fbn3=fbn3,cln3=cln3,
			nrecimg2=nrecimg2,nrec2=nrec2,egn2 =egn2,cbn2 =cbn2,prn2 =prn2,ftn2=ftn2,fbn2=fbn2,cln2=cln2,
			nrecimg3=nrecimg3,nrec3=nrec3,egn1=egn3,cbn1 =cbn3,prn1 =prn1,ftn1=ftn1,fbn1=fbn1,cln1=cln1,
			rrecimg1=rrecimg1,rrec1=rrec1,egr3 =egr3,cbr3 =cbr3,prr3 =prr3,ftr3=ftr3,fbr3=fbr3,clr3=clr3,
			rrecimg2=rrecimg2,rrec2=rrec2,egr2 =egr2,cbr2 =cbr2,prr2 =prr2,ftr2=ftr2,fbr2=fbr2,clr2=clr2,
			rrecimg3=rrecimg3,rrec3=rrec3,egr1=egr1, cbr1 =cbr1,prr1 =prr1,ftr1=ftr1,fbr1=fbr1,clr1=clr1)
# No caching at all for API endpoints.
'''
@app.after_request
def after_request(response):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
        response.headers["Expires"] = 0
        response.headers["Pragma"] = "no-cache"
        return response
'''
@app.after_request
def add_header(response):
    response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


if __name__ == '__main__':
	# load ml model
	model = load_model('model20 no aug 81.hdf5')
	model._make_predict_function()
	# start api
	app.run(host='0.0.0.0', port=5001, debug=True)
