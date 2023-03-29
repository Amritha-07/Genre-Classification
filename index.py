from flask import Flask,render_template,request
import pickle
import sklearn
import numpy as np
import statistics

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
	return render_template('Home.html')
	# return "hello"
@app.route('/predict', methods = ['POST'])
def predict():
	if request.method == 'POST':
		lyric = request.form["lyric"]
		cv = pickle.load(open("GenreClassificationCV.pkl", "rb"))
		models = pickle.load(open("GenreClassificationModel.pkl", "rb"))
		inputs = cv.transform([(lyric)])
		arr = []
		for model in models.keys():
			pred = models[model].predict(inputs)[0]
			arr.append(pred)
		result = "Lyric Genre: " + statistics.mode(arr)
		return render_template("Home.html", a = result)

if __name__ == '__main__':
	app.debug = True
	app.run(host = "0.0.0.0", port = 5000)
