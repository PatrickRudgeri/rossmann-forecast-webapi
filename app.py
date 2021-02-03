from flask import Flask, request, Response, render_template
from pipeline import ProductionPipeline
import joblib
import pandas as pd
import os

root_path = ""
app = Flask(__name__, template_folder=root_path + "templates")

# Carregando modelo em memória apenas uma vez (quando a API iniciar)
model_path = root_path + "models/XGBoost_model_2.pkl"

model = joblib.load(model_path)
    

@app.route("/")
def main():
    return render_template("main.html")


@app.route("/rossmann/predict", methods=["post"])
def rossmann_predict():
    test_json = request.get_json() 
    # json deve vir no formato "records", onde cada linha do dataframe é um dict,
    # caso seja mais de uma linha deve vir como uma lista de dicts
    
    if test_json:
        if isinstance(test_json, dict): 
            # unique example
            df_raw = pd.DataFrame(test_json, index=[0])
            
        else: 
            # multiple examples
            df_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
    
        pipeline = ProductionPipeline()

        # aplicar limpezas, transformações e encoders
        df = pipeline.start_pipeline(df_raw)

        # prediction
        # `df_raw` é necessário para retornar os dados originais
        response = pipeline.get_prediction(model, df_raw, df) 
        
        return response
        
        
    else:
        # se a requisição vier vazia, então responder com "{}" (vazio) e retornar status 200
        return Response("{}", status=200, mimetype="application/json")
    
if __name__ == "__main__":
    port = os.environ.get("PORT", 5000)
    host = "0.0.0.0"
    app.run(host=host, port=port)