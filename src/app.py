import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request

import utils


project_dir = Path(__file__).resolve().parents[1]
app = Flask(__name__)


with open(project_dir / "models" / "model.pickle", "rb") as f:
    model = pickle.load(f)
with open(project_dir / "models" / "encoder.pickle", "rb") as f:
    enc = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    a = request.form.to_dict()
    print("-----------------------------------")
    print(a)
    print("___________________________")
    ################################
    b={'age_of_driver':a['age_of_driver'],'gender':a['gender'],'marital_status':a['marital_status'],'safty_rating':a['safty_rating'],'annual_income':a['annual_income'],'high_education_ind':a['high_education_ind'],'address_change_ind':a['address_change_ind'],'living_status':a['living_status'], 'accident_site':a['accident_site'],'past_num_of_claims':a['past_num_of_claims'],'witness_present_ind':a['witness_present_ind'], 'liab_prct':a['liab_prct'], 'channel':a['channel'],'policy_report_filed_ind':a['policy_report_filed_ind'],'claim_est_payout':a['claim_est_payout'], 'age_of_vehicle':a['age_of_vehicle'],'vehicle_category':a['vehicle_category'], 'vehicle_price':a['vehicle_price'], 'vehicle_weight':a['vehicle_weight'], 'latitude':88.8,'longitude':121.2}
    a1=pd.DataFrame(b,index=[0])
    t=enc.transform(a1)
    p=model.predict(t)
    ##############################
    #preprocessed_data = utils.preprocess(form_data)
    print("Result",p)
    #probability = model.predict_proba(pd.DataFrame(preprocessed_data, index=[0]))
    return render_template("result.html", probability=p)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081)
