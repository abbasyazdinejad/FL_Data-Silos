import pandas as pd
import numpy as np

# Simulated settings
scenarios = [
    "Centralized AI",
    "Federated (No Privacy)",
    "Federated + DP (ε ≈ 3.0)",
    "Federated + DP (ε ≈ 1.5)",
    "Federated + DP (ε ≈ 1.0)",
]

# Use plausible values based on your paper
data = {
    "Scenario": scenarios,
    "Accuracy (%)": [95.9, 94.2, 92.5, 91.8, 91.3],
    "Macro-F1 (%)": [94.1, 92.6, 91.4, 91.2, 91.1],
    "Estimated ε": [np.inf, np.inf, 3.0, 1.5, 1.0]
}

df = pd.DataFrame(data)

# Optional: round and format
df["Accuracy (%)"] = df["Accuracy (%)"].map(lambda x: f"{x:.1f}")
df["Macro-F1 (%)"] = df["Macro-F1 (%)"].map(lambda x: f"{x:.1f}")
df["Estimated ε"] = df["Estimated ε"].map(lambda x: f"{x}" if x != np.inf else "∞")

# Show the simulated results
print("🔐 Simulated FL + Privacy Performance Summary")
print(df.to_markdown(index=False))
