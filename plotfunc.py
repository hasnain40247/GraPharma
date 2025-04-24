import pandas as pd
import matplotlib.pyplot as plt

# Load performance log files
ffn_df = pd.read_csv("./metrics/ffn_performance_log.csv")
gnn_df = pd.read_csv("./metrics/gnn_performance_log.csv")
lstm_df = pd.read_csv("./metrics/lstm_performance_log.csv")

# Extract final test RMSE from each model type
ffn_rmse = {"FFN": ffn_df["test_rmse"].iloc[-1]}
gnn_rmse = gnn_df.groupby("architecture")["test_rmse"].apply(lambda x: x.iloc[-1]).to_dict()
lstm_rmse = lstm_df.groupby("model_type")["test_rmse"].apply(lambda x: x.iloc[-1]).to_dict()

# Combine all RMSE values
all_rmse = {**ffn_rmse, **gnn_rmse, **lstm_rmse}

# Sort by RMSE for clear visualization
sorted_rmse = dict(sorted(all_rmse.items(), key=lambda item: item[1]))

# Plotting
plt.figure(figsize=(10, 6))
bars = plt.bar(sorted_rmse.keys(), sorted_rmse.values(), color='lightsteelblue')
plt.ylabel("Final Test RMSE")
plt.title("Final Test RMSE Comparison Across All Model Types")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Annotate RMSE values on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.02, f"{height:.2f}", ha='center', va='bottom')

plt.tight_layout()
plt.savefig("./metrics/final_test_rmse_comparison.png", dpi=300)  # optional: save as PNG
plt.show()
