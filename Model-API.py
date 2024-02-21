from flask import Flask, request, redirect
from flask_restful import Resource, Api
from flask_cors import CORS
import os
import sys
import predictions
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, jsonify
from sklearn.cluster import KMeans
import joblib
import your_ml_model

app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})
api = Api(app)

# input_data = {
#      'BMC step type': ["REVENUE STREAMS"],
#      'Problem confident': ["Fair"],
#      'Problem current progress': ["In progress - 20%"],
#      'Problem level support':["Full support required	"],
#      'Problem rate': [2],  
#      'Creator': ["hjghjkjdff@gmail.com"]
#      }

class Test(Resource):
    def get(self):
        return 'Welcome to, Test App API!'

    def post(self):
        try:
            value = request.get_json()
            if(value):
                return {'Post Values': value}, 201

            return {"error":"Invalid format."}

        except Exception as error:
            return {'error': error}

class GetPredictionOutput(Resource):
    def get(self):
        return {"error":"Invalid Method."}

    def post(self):
        try:

            data = request.get_json()
            # predict = predictions.predict_mpg(data)
            # predictOutput = predict
            # return jsonify({'predict':predictOutput})
            df=pd.read_csv(r"C:\Users\Hamza\Downloads\Simsy-data.csv",encoding="latin1")
            df=df[["BMC step type","Problem confident","Problem current progress","Problem ideas","Problem level support","Problem rate","Creator"]]
            df1=df.dropna()
            df1.drop("Problem ideas",axis=1,inplace=True)
            input_df=pd.DataFrame(data)
            combined_df = pd.concat([df1, input_df], ignore_index=True)
            cat_columns = ['BMC step type', 'Problem confident', 'Problem current progress', 'Problem level support']
            one_hot_encoded = pd.get_dummies(combined_df[cat_columns])
            combined_data = pd.concat([one_hot_encoded, combined_df[['Problem rate']]], axis=1)
            scaler = StandardScaler()
            numerical_data = scaler.fit_transform(combined_data)
            warnings.filterwarnings(action='ignore')
            wcss = []
            scaled_data=numerical_data
            for i in range(1, 11):  
                kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
                kmeans.fit(scaled_data)
                wcss.append(kmeans.inertia_)
    # Load the model using joblib
            num_clusters = 10  
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(numerical_data[:len(numerical_data)-1])
            loaded_model = joblib.load(r'C:\Users\Hamza\Downloads\KMeans.pkl')
            input_cluster = cluster_labels[-1]
            similar_indices = [i for i, label in enumerate(cluster_labels) if label == input_cluster]
            similarities = cosine_similarity(numerical_data[-1].reshape(1, -1), numerical_data[similar_indices])
            most_similar_index = similar_indices[similarities.argmax()]

            # most_similar_row = df1.iloc[most_similar_index]

    # print("Most similar row:")
    # print(most_similar_row)
            return jsonify({'user': most_similar_index})

        except Exception as e:
        # Handle the UnsupportedMediaType error
            return jsonify({'error': 'Unsupported media type'}), 415

api.add_resource(Test,'/')
api.add_resource(GetPredictionOutput,'/getPredictionOutput')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)