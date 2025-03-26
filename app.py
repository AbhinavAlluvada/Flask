#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, render_template
import pickle
import numpy as np

with open('hierarchical_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

@app.route('/home')
def home_page():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = request.form['features']
        features = np.array([float(x) for x in features.split(',')]).reshape(1, -1)
        cluster_label = model.fit_predict(features)[0]
        return render_template("result.html", cluster=int(cluster_label))
    except Exception as e:
        return render_template("result.html", cluster="Error")

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




