# Solar PV Live Forecast — Streamlit Community Cloud

## Deployment requirement
This project must run on **Python 3.12** (recommended) or **3.13**. The saved `.keras` models use Keras 3 and require a TensorFlow build available for the selected Python runtime.

### Streamlit Community Cloud
1. Put the contents of this folder at the **root of your GitHub repository**. `app.py`, `requirements.txt`, and `model_bundle/` must be siblings.
2. Create a **new** Streamlit app (do not reuse an app that was previously deployed with Python 3.14).
3. Open **Advanced settings** and select **Python 3.12**.
4. Set `app.py` as the main file.
5. Deploy.

> Important: Streamlit Community Cloud does not change the Python runtime of an already-deployed app when `requirements.txt` changes. To change Python, delete the old app and redeploy it with the desired Python version.

## Google Sheet
The app first tries the public Google Sheets CSV endpoint. The worksheet must contain `Date` and `Time` columns plus the model feature columns in `model_bundle/configuration/manifest.json`.

For a private sheet, add `gcp_service_account` in Streamlit Secrets.

## Forecast
- Uses all usable observations read from the configured Google Sheet.
- Uses the saved Q-learning model scores.
- Uses the saved uncertainty-aware dynamic stacker.
- Produces recursive H+1 ... H+24 forecasts at 5-minute spacing.
- Only exposes forecast points inside the configured 06:00–18:00 operating window.
- Does not retrain models on the cloud.
