# Industrial Machine Learning - Predictive Maintenance

Predicts whether an industrial machine will fail from sensor readings
(temperature, vibration, pressure, rotation speed, operating hours).

## Setup
    pip install -r requirements.txt

## Run
    python industrial_ml.py

## Output
- Console: dataset summary, classification report, confusion matrix, ROC-AUC
- feature_importance.png: which sensors matter most
- maintenance_model.joblib: saved trained model
- A failure prediction for a new machine reading

## Use your own data
In industrial_ml.py, replace `df = generate_sensor_data()` with
`df = pd.read_csv("your_data.csv")` (needs a binary `failure` column).
