import streamlit as st
import pandas as pd
import numpy as np
import warnings

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Safe imports with error handling
try:
    import shap
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import spacy
    ML_AVAILABLE = True
except ImportError as e:
    st.error(f"Missing ML dependencies: {e}")
    ML_AVAILABLE = False

# ===============================
# 0. NLP SETUP
# ===============================
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_md")
    except OSError:
        st.error("spaCy model not found. Run: python -m spacy download en_core_web_md")
        st.stop()

nlp_model = load_spacy_model()

# ===============================
# 1. DATA & MODEL SETUP
# ===============================
@st.cache_resource # This keeps the model and scaler in memory
def load_and_train():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    df = pd.read_csv(url, names=names, na_values="?").dropna()
    df['target'] = (df['target'] > 0).astype(int)

    # Identify categorical columns for one-hot encoding
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=bool)

    # Separate features (X) and target (y)
    X = df_encoded.drop('target', axis=1)
    y = df_encoded['target']

    # Get feature names after one-hot encoding
    feature_columns = X.columns

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Model on scaled data
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    # Initialize SHAP explainer with the trained model and scaled training data
    explainer = shap.TreeExplainer(model, X_scaled)

    return model, explainer, feature_columns, scaler

model, explainer, feature_names, scaler = load_and_train()

# ===============================
# 2. STREAMLIT UI
# ===============================
st.title("❤️ Heart Disease Predictor & Explainer")

# # Ngrok tunnel setup
# try:
#     from ngrok import start_ngrok_tunnel
#     public_url = start_ngrok_tunnel(8501)  # Use port 8501 to match Streamlit default
#     st.success(f"🌐 **Public URL:** {public_url}")
#     st.markdown(f"[Open in Browser]({public_url})")
# except Exception as e:
#     st.warning(f"Ngrok not available: {e}")

st.sidebar.header("Patient Input Data")

def user_input_features():
    age = st.sidebar.slider('Age', 20, 80, 50)
    sex = st.sidebar.selectbox('Sex (1=Male, 0=Female)', [1.0, 0.0]) # Ensure float type
    cp = st.sidebar.selectbox('Chest Pain Type (0:Typical Angina, 1:Atypical Angina, 2:Non-anginal Pain, 3:Asymptomatic)', [0.0, 1.0, 2.0, 3.0], index=1) # Ensure float type
    trestbps = st.sidebar.slider('Resting Blood Pressure (mm Hg)', 90, 200, 120)
    chol = st.sidebar.slider('Serum Cholestoral (mg/dl)', 120, 560, 200)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)', [0.0, 1.0]) # Ensure float type
    restecg = st.sidebar.selectbox('Resting Electrocardiographic Results (0:Normal, 1:ST-T wave abnormality, 2:Left ventricular hypertrophy)', [0.0, 1.0, 2.0], index=1) # Ensure float type
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved', 70, 200, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina (1=Yes, 0=No)', [0.0, 1.0]) # Ensure float type
    oldpeak = st.sidebar.slider('ST Depression Induced by Exercise Relative to Rest', 0.0, 6.2, 1.0)
    slope = st.sidebar.selectbox('Slope of the Peak Exercise ST Segment (0:Upsloping, 1:Flat, 2:Downsloping)', [0.0, 1.0, 2.0], index=1) # Ensure float type
    ca = st.sidebar.selectbox('Number of Major Vessels Colored by Flourosopy (0-3)', [0.0, 1.0, 2.0, 3.0]) # Ensure float type
    thal = st.sidebar.selectbox('Thal (3:Normazl, 6:Fixed Defect, 7:Reversable Defect)', [3.0, 6.0, 7.0], index=0) # Ensure float type


    notes = st.sidebar.text_area(
        "Clinical Notes",
        "Patient reports chest tightness during exertion."
    )

    raw = pd.DataFrame({
        'age':[age],'sex':[sex],'cp':[cp],'trestbps':[trestbps],
        'chol':[chol],'fbs':[fbs],'restecg':[restecg],
        'thalach':[thalach],'exang':[exang],'oldpeak':[oldpeak],
        'slope':[slope],'ca':[ca],'thal':[thal]
    })

    categorical = ['sex','cp','fbs','restecg','exang','slope','ca','thal']
    encoded = pd.get_dummies(raw, columns=categorical, drop_first=True, dtype=bool)

    # Create processed DataFrame with all expected features
    processed = pd.DataFrame(columns=feature_names)
    
    # First, add all encoded columns that exist
    for col in feature_names:
        if col in encoded.columns:
            processed[col] = encoded[col].values
        else:
            processed[col] = False
    
    # Then, add numerical columns
    numerical_cols = ['age','trestbps','chol','thalach','oldpeak']
    for col in numerical_cols:
        if col in raw.columns:
            processed[col] = raw[col].values

    return processed, notes

input_df, clinical_notes = user_input_features()

# ===============================
# 3. PREDICTION
# ===============================
st.subheader("Processed Patient Data")
st.write(input_df)

input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)
prob = model.predict_proba(input_scaled)

st.subheader("Prediction Result")
risk_level = "High Risk" if prob[0][1] > 0.5 else "Medium Risk" if prob[0][1] > 0.3 else "Low Risk"
st.metric("Diagnosis", risk_level, f"{prob[0][1]*100:.1f}% Probability")

# ===============================
# SHAP EXPLANATION (FINAL FIX)
# ===============================
st.subheader("Why this prediction?")

shap_values = explainer.shap_values(input_scaled)

# Binary classifier → take positive class
if isinstance(shap_values, list):
    shap_vals = shap_values[1]
else:
    shap_vals = shap_values

# Convert safely to 1D
shap_1d = np.asarray(shap_vals).flatten()

# 🔐 HARD SAFETY CHECK
min_len = min(len(feature_names), len(shap_1d))

feature_importance = pd.DataFrame({
    "feature": feature_names[:min_len],
    "importance": shap_1d[:min_len]
}).sort_values("importance", key=abs)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(feature_importance["feature"], feature_importance["importance"])
ax.axvline(0, color="red", linestyle="--")
ax.set_xlabel("SHAP Value (Impact)")
ax.set_title("Feature Contribution to Prediction")
plt.tight_layout()
st.pyplot(fig)

# ===============================
# 5. NLP CLINICAL ENTITY EXTRACTION
# ===============================
st.subheader("NLP-Extracted Clinical Entities")
st.write(f"**Clinical Notes:** {clinical_notes}")

doc = nlp_model(clinical_notes)

# Enhanced medical entity detection with keyword matching
medical_keywords = {
    "chest tightness": "SYMPTOM",
    "chest pain": "SYMPTOM", 
    "shortness of breath": "SYMPTOM",
    "dizziness": "SYMPTOM",
    "sweating": "SYMPTOM",
    "heart disease": "CONDITION",
    "heart attack": "CONDITION",
    "coronary artery disease": "CONDITION",
    "high blood pressure": "CONDITION",
    "hypertension": "CONDITION",
    "diabetes": "CONDITION",
    "exertion": "ACTIVITY",
    "family history": "RISK_FACTOR",
    "radiating": "SYMPTOM_DESCRIPTION"
}

found_entities = False

# Check spaCy entities first
if doc.ents:
    for ent in doc.ents:
        st.write(f"- **{ent.text}** → {ent.label_}")
        found_entities = True

# Add keyword-based medical entity detection
for keyword, entity_type in medical_keywords.items():
    if keyword.lower() in clinical_notes.lower():
        st.write(f"- **{keyword}** → {entity_type}")
        found_entities = True

if not found_entities:
    st.write("No medical entities detected. Try terms like: chest pain, shortness of breath, heart disease")
