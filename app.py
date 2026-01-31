# import streamlit as st
# import joblib
# import numpy as np

# # load trained regression model
 
# model = joblib.load('regression_model.joblib')

# # app UI
# st.title("Job Package Prediction Based on CGPA")
# st.write("Enter your CGPA to predict the expected job package:")

# # CGPA input
# cgpa = st.number_input(
#     "CGPA",
#     min_value=0.0,
#     max_value=10.0,
#     step=0.1
# )

# # predict button

# if st.button("Predict Package"):
#     # prepare input for model
#     input_data = np.array([[cgpa]])

#     # make prediction
#     prediction = model.predict(input_data)

#     # Convert Numpy output to Python float safely
#     predicted_value = prediction.item()

#     # Optional: prevent negative output
#     predicted_value = max(predicted_value, 0)

#     # display result
#     st.success(f"Predicted Package:₹{predicted_value:,.2f} LPA")

import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Job Package Predictor",
    page_icon="💼",
    layout="centered"
)

# ------------------ Load Model ------------------
model = joblib.load("regression_model.joblib")

# ------------------ Title ------------------
st.title("💼 Job Package Prediction Based on CGPA")
st.write("Drag the slider to instantly predict your job package")

# ------------------ CGPA Slider (Auto Trigger) ------------------
cgpa = st.slider(
    "🎓 Select your CGPA",
    min_value=0.0,
    max_value=10.0,
    value=7.0,
    step=0.1
)

# ------------------ Prediction (Auto) ------------------
input_data = np.array([[cgpa]])
predicted_value = max(model.predict(input_data).item(), 0)

# ------------------ Display Result ------------------
st.success(f"💰 Predicted Job Package: ₹ {predicted_value:,.2f} LPA")

# ------------------ Graph ------------------
st.markdown("### 📊 CGPA vs Job Package Trend")

# Generate CGPA range
cgpa_range = np.linspace(0, 10, 100).reshape(-1, 1)
package_predictions = model.predict(cgpa_range).flatten()

# Plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(cgpa_range, package_predictions, label="Prediction Trend")
ax.scatter(cgpa, predicted_value, s=120, label="Your CGPA")

ax.set_xlabel("CGPA")
ax.set_ylabel("Package (LPA)")
ax.set_title("Relationship Between CGPA and Job Package")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# ------------------ Footer ------------------
st.markdown("---")
st.caption("📈 Machine Learning Project | Live Prediction with Streamlit")
