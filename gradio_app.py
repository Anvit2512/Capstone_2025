import gradio as gr
import torch
import torch.nn as nn
import json
import numpy as np
import joblib
import re
from torch.nn.utils.parametrizations import weight_norm

# --------------------------------------------------
# 1. Model Architecture (Must match your training code exactly)
# --------------------------------------------------
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(TCNBlock, self).__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                                         stride=1, padding=padding, dilation=dilation))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.chomp = lambda x: x[:, :, :-padding] if padding > 0 else x
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.net = nn.Sequential(self.conv, self.relu, self.dropout)

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        out = self.chomp(self.net(x))
        return self.relu(out + res)

class TCNModel(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=3, output_size=1):
        super(TCNModel, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_ch = input_size if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation))
        self.network = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.network(x)
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)

# --------------------------------------------------
# 2. Initialization: Load Model and Scaler
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Scaler
scaler = joblib.load("scaler.pkl")

# Load Model
model = TCNModel(input_size=1, num_channels=[32, 64, 128]).to(device)
model.load_state_dict(torch.load("best_capstone_model.pth", map_location=device))
model.eval()

# --------------------------------------------------
# 3. Prediction Function
# --------------------------------------------------
def predict(json_input):
    try:
        # Parse the JSON string
        data = json.loads(json_input)

        # Extract 'v' (adc) values. Handle list or single dict.
        if isinstance(data, dict):
            data = [data]

        adc_values = np.array([float(item['v']) for item in data]).reshape(-1, 1)

        # 1. Scale the input using the saved scaler
        scaled_values = scaler.transform(adc_values)

        # 2. Convert to Tensor [Batch=1, TimeSteps, Features=1]
        input_tensor = torch.FloatTensor(scaled_values).unsqueeze(0).to(device)

        # 3. Model Inference
        with torch.no_grad():
            prediction = model(input_tensor)

        result = prediction.item()
        return f"Prediction: {result:.4f}"

    except json.JSONDecodeError:
        return "Error: Invalid JSON format. Please check your brackets and commas."
    except KeyError:
        return "Error: JSON objects must contain a 'v' key for ADC values."
    except Exception as e:
        return f"Error: {str(e)}"

# --------------------------------------------------
# 4. Gradio Interface
# --------------------------------------------------
example_json = """[
  {"t": 0, "v": 512},
  {"t": 12, "v": 530},
  {"t": 25, "v": 548}
]"""

ui = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        lines=10,
        placeholder="Enter JSON here...",
        label="Input JSON (ADC Sequence)",
        value=example_json
    ),
    outputs=gr.Textbox(label="Model Prediction Output"),
    title="ADC Time-Series Predictor",
    description="Paste a JSON array of ADC values. The model uses the 'v' key for prediction."
)

if __name__ == "__main__":
    ui.launch()
