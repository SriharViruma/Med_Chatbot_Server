from flask import Flask
from flask import request
import psycopg2
import json
user = "koztiptegxtyxo"
dbpassword = "fd2a7cc709bc7f38cb7923954b139caa82a5a9bf4509a04c3ad86576497dff93"
host = "ec2-50-17-21-170.compute-1.amazonaws.com"
port = "5432"
database = "d8ffr2oj2m71si"
app = Flask(__name__)
#implemented with flask 
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tflearn
import tensorflow
import random
import json
import pickle
with open("intents.json") as file:
	data = json.load(file)
try:
	ds
	with open("data.pickle", "rb") as f:
		words, labels, training, output = pickle.load(f)
except:
	words = []
	labels = []
	docs_x = []
	docs_y = []
	for intent in data["intents"]:
		for pattern in intent["patterns"]:
			wrds = nltk.word_tokenize(pattern)
			words.extend(wrds)
			docs_x.append(wrds)
			docs_y.append(intent["tag"])
			if intent["tag"] not in labels:
				labels.append(intent["tag"])
	words = [stemmer.stem(w.lower()) for w in words if w != "?"]
	words = sorted(list(set(words)))
	labels = sorted(labels)
	training = []
	output = []
	out_empty = [0 for _ in range(len(labels))]
	for x, doc in enumerate(docs_x):
		bag = []
		wrds = [stemmer.stem(w.lower()) for w in doc]
		for w in words:
			if w in wrds:
				bag.append(1)
			else:
				bag.append(0)
		output_row = out_empty[:]
		output_row[labels.index(docs_y[x])] = 1
		training.append(bag)
		output.append(output_row)
	training = numpy.array(training)
	output = numpy.array(output)
	with open("data.pickle", "wb") as f:
		pickle.dump((words, labels, training, output), f)
tensorflow.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)
try:
	
	model.load("model.tflearn")
except:
	model = tflearn.DNN(net)
	model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
	model.save("model.tflearn")
def bag_of_words(s, words):
	bag = [0 for _ in range(len(words))]
	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]
	for se in s_words:
		for i, w in enumerate(words):
			if w == se:
				bag[i] = 1
	return numpy.array(bag)

@app.route('/chatbot/<name>')
def hello_world(name):
	email = request.args.get('email')
	reply = chat(name)
	result = None
	details = None
	
	try:
		connection = psycopg2.connect(user=user,password=dbpassword,host=host,port=port,database=database)
		cursor = connection.cursor()
		select_query = "SELECT *from DISEASES where DISEASE= %s"
		cursor.execute(select_query, (reply,))
		result = cursor.fetchall();
		if len(result)>0:
			print(reply,email)
			update_query = """UPDATE userdata SET cd=%s WHERE email=%s"""
			cursor.execute(update_query,(reply,email))
			updated_rows = cursor.rowcount
			connection.commit()
			print(updated_rows)
	except (Exception, psycopg2.DatabaseError) as error:
		print(error)
	finally:
		if(connection):
			cursor.close()
			connection.close()
			print("PostgreSQL connection is closed")
			details = {'reply':reply}
			return json.dumps(details)

@app.route('/')
def hel():
    return 'Home route'
@app.route('/register/')
def reg():
	try:
		email = request.args.get('email')
		name = request.args.get('name')
		password = request.args.get('password')
		height = request.args.get('height')
		weight= request.args.get('weight')
		gender = request.args.get('gender')
		age = request.args.get('age')
		gp = request.args.get('gp')
		connection = psycopg2.connect(user=user,password=dbpassword,host=host,port=port,database=database)
		cursor = connection.cursor()
		postgres_insert_query = """ INSERT INTO userdata (EMAIL, NAME, PASSWORD, HEIGHT, WEIGHT, GENDER, AGE, GP, CD) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
		record_to_insert = (email, name, password, int(height), int(weight), gender, int(age), gp, 'none')
		cursor.execute(postgres_insert_query, record_to_insert)
		connection.commit()
		count = cursor.rowcount
		print (count, "Record inserted successfully into mobile table")
		if(connection):
			print("Failed to insert record into mobile table", error)
	finally:
		if(connection):
			cursor.close()
			connection.close()
			print("PostgreSQL connection is closed")
		return 'Successfully registered'

@app.route('/login/')
def log():
	email = request.args.get('email')
	password = request.args.get('password')
	try:
		connection = psycopg2.connect(user=user,password=dbpassword,host=host,port=port,database=database)
		cursor = connection.cursor()
		select_query = "SELECT PASSWORD from USERDATA where EMAIL= %s"
		cursor.execute(select_query, (email,))
		result = cursor.fetchall();
		rows = cursor.rowcount
		print (rows)
	finally:
		if(connection):
			cursor.close()
			connection.close()
			print("PostgreSQL connection is closed")
			if rows == 1 and result[0][0] == password:
			        return email
			else:
					return "error"

@app.route('/doEdit/')
def edit():
	email = request.args.get('email')
	age = request.args.get('age')
	height = request.args.get('height')
	weight= request.args.get('weight')
	gp = request.args.get('gp')
	try:
		details = None
		connection = psycopg2.connect(user=user,password=dbpassword,host=host,port=port,database=database)
		cursor = connection.cursor()
		update_query = """UPDATE userdata SET age=%s,height=%s,weight=%s,gp=%s WHERE email=%s"""
		cursor.execute(update_query,(age,height,weight,gp,email))
		updated_rows = cursor.rowcount
		connection.commit()
		select_query = "SELECT * from USERDATA where EMAIL= %s"
		cursor.execute(select_query, (email,))
		result = cursor.fetchall();
		details={'name':result[0][1],'age':str(result[0][6]),'gender':result[0][5],'height':str(result[0][3]),'weight':str(result[0][4]),'gp':result[0][7],'cd':result[0][8]}
		print(result)
	finally:
		if(connection):
			cursor.close()
			connection.close()
			print("PostgreSQL connection is closed")
			return json.dumps(details)

@app.route('/getDetails/')
def det():
	email = request.args.get('email')
	try:
		details = None
		connection = psycopg2.connect(user=user,password=dbpassword,host=host,port=port,database=database)
		cursor = connection.cursor()
		select_query = "SELECT * from USERDATA where EMAIL= %s"
		cursor.execute(select_query, (email,))
		result = cursor.fetchall();
		details={'name':result[0][1],'age':str(result[0][6]),'gender':result[0][5],'height':str(result[0][3]),'weight':str(result[0][4]),'gp':result[0][7],'cd':result[0][8]}
		print(result)
	finally:
		if(connection):
			cursor.close()
			connection.close()
			print("PostgreSQL connection is closed")
			return json.dumps(details)

@app.route('/getSuggestions/')
def sug():
	email = request.args.get('email')
	try:
		details = None
		connection = psycopg2.connect(user=user,password=dbpassword,host=host,port=port,database=database)
		cursor = connection.cursor()
		select_user = "SELECT CD from USERDATA where EMAIL= %s"
		cursor.execute(select_user, (email,))
		result = cursor.fetchall();
		disease = result[0][0]
		select_disease = "SELECT *from DISEASES where DISEASE= %s"
		cursor.execute(select_disease, (disease,))
		suggestion = cursor.fetchall();
		details = {'disease':disease,'diet':suggestion[0][1],'lat':suggestion[0][2],'lon':suggestion[0][3]}
		print (suggestion)
	finally:
		if(connection):
			cursor.close()
			connection.close()
			print("PostgreSQL connection is closed")
			return json.dumps(details) 

def chat(inp):
	results = model.predict([bag_of_words(inp, words)])[0]
	results_index = numpy.argmax(results)
	tag = labels[results_index]
	if results[results_index] > 0.7:
	    for tg in data["intents"]:
	    	if tg['tag'] == tag:
	    		responses = tg['responses']
	    return random.choice(responses)
	else:
	    return "Sorry I don't understand"

if __name__ == '__main__':
   app.run(debug= True)


