import streamlit as st
import numpy as np

# Constants
SEQ_LEN = 50
FEATURES = ['HR', 'Chest EDA', 'Wrist EDA', 'ECG']

# Initialize session buffer
if 'buffer' not in st.session_state:
    st.session_state.buffer = np.zeros((SEQ_LEN, 4))  # shape: (time, features)
    st.session_state.spike_timer = 0

#adding slider and button
st.sidebar.header("Stress Spike Settings")
hr_spike = st.sidebar.slider("HR Spike (bpm)", 100, 150, 120)
chest_eda_spike = st.sidebar.slider("Chest EDA Spike", 0.8, 1.8, 1.2)
wrist_eda_spike = st.sidebar.slider("Wrist EDA Spike", 0.8, 1.8, 1.2)
ecg_spike = st.sidebar.slider("Chest ECG Spike", 1.0, 1.4, 1.2)

if st.sidebar.button("Trigger Stress Spike"):
    st.session_state.spike_timer = 10  # next 10 timesteps will be spiked

# Generate one new sample (spiked or normal)
if st.session_state.spike_timer > 0:
    new_sample = np.array([hr_spike, chest_eda_spike, wrist_eda_spike, ecg_spike])
    st.session_state.spike_timer -= 1
else:
    new_sample = np.array([
        np.random.normal(75, 4),     # HR
        np.random.normal(0.4, 0.05), # Chest EDA
        np.random.normal(0.45, 0.05),# Wrist EDA
        np.random.normal(1.0, 0.05)  # ECG
    ])

# Update buffer (rolling window)
st.session_state.buffer = np.vstack([st.session_state.buffer[1:], new_sample])

import matplotlib.pyplot as plt

fig, axs = plt.subplots(4, 1, figsize=(8, 6), sharex=True)
for i, feature in enumerate(FEATURES):
    axs[i].plot(st.session_state.buffer[:, i])
    axs[i].set_ylabel(feature)
    axs[i].grid(True)
axs[-1].set_xlabel("Timestep")
st.pyplot(fig)


### Getting the model weights
import torch
import torch.nn as nn

# Define your model architecture again
class StressLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=4, output_size=1):
        super(StressLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # Use the last time step

# Load the model
model = StressLSTM()
model.load_state_dict(torch.load("LSTM_weights", map_location=torch.device("cpu")))
model.eval()


# Prepare input: shape (1, seq_len, 4)
input_tensor = torch.tensor(st.session_state.buffer, dtype=torch.float32).unsqueeze(0)

# Predict
with torch.no_grad():
    logits = model(input_tensor)
    probs = torch.softmax(logits, dim=1).squeeze()

# Get prediction
logits = model(input_tensor)
probs = torch.sigmoid(logits).squeeze()  # shape: scalar (0-dim tensor)
confidence = probs.item()
pred_class = 1 if confidence > 0.5 else 0
label_map = {0: "Baseline", 1: "Stress"}

st.markdown("### 🤖 Model Prediction")
st.metric("Predicted State", label_map[pred_class], f"{confidence*100:.1f}% confidence")


# Optional debug info
with st.expander("Show Raw Probability"):
    st.write(f"Probability of Stress (class 1): {confidence:.3f}")


