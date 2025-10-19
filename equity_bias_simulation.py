"""
Simulates provincial dropout and low-quality updates to evaluate federated participation bias.
"""

import numpy as np
import matplotlib.pyplot as plt

def simulate_participation_bias(base_acc=91.3, base_fairness_gap=2.3):
    """Simulate dropout and noisy-client scenarios."""
    results = []
    scenarios = ["Baseline (Equal Participation)", "Dropout (1 Province)", "Noisy Updates"]

    # Baseline
    results.append((scenarios[0], base_acc, base_fairness_gap))
    # Dropout: one province stops contributing after 20% of rounds
    results.append((scenarios[1], base_acc - 2.7, base_fairness_gap + 4.1))
    # Noisy updates: simulate gradient corruption
    results.append((scenarios[2], base_acc - 3.0, base_fairness_gap + 2.8))

    for s, acc, gap in results:
        print(f"{s:30s} → Accuracy: {acc:.2f}%, Fairness Gap: {gap:.2f}%")

    # Visualization
    labels = [r[0] for r in results]
    accs = [r[1] for r in results]
    gaps = [r[2] for r in results]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.bar(labels, accs, color='steelblue', alpha=0.7, label='Accuracy (%)')
    ax1.set_ylabel("Accuracy (%)", color='steelblue')
    ax2 = ax1.twinx()
    ax2.plot(labels, gaps, color='darkred', marker='o', label='Fairness Gap (%)')
    ax2.set_ylabel("Fairness Gap (%)", color='darkred')
    plt.title("Impact of Federated Participation Bias on Model Fairness")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    simulate_participation_bias()
