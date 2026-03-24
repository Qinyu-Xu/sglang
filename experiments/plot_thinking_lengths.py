import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = json.load(open("data/results/thinking_lengths.json"))
chat_out      = np.array(data["chat_out"])
reasoning_out = np.array(data["reasoning_out"])
reasoning_think = np.array(data["reasoning_think"])

all_out = np.concatenate([chat_out, reasoning_out])
bins = np.linspace(0, np.percentile(all_out, 99), 60)

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(chat_out,      bins=bins, color="#4C72B0", alpha=0.6, label=f"chat (no thinking)  median={np.median(chat_out):.0f}")
ax.hist(reasoning_out, bins=bins, color="#DD8452", alpha=0.6, label=f"reasoning (AIME)    median={np.median(reasoning_out):.0f}")
for arr, color in [(chat_out, "#4C72B0"), (reasoning_out, "#DD8452")]:
    ax.axvline(np.median(arr), color=color, linestyle="--", linewidth=1.5)
ax.set_title(f"Qwen3 output tokens — chat vs reasoning  (n={len(chat_out)} each)\nMedian ratio: {np.median(reasoning_out)/max(np.median(chat_out),1):.1f}×")
ax.set_xlabel("output tokens")
ax.set_ylabel("count")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("data/results/thinking_lengths.png", dpi=150)
print("saved → data/results/thinking_lengths.png")

print(f"\nchat      — median {np.median(chat_out):.0f}  mean {np.mean(chat_out):.0f}  p95 {np.percentile(chat_out,95):.0f}")
print(f"reasoning — median {np.median(reasoning_out):.0f}  mean {np.mean(reasoning_out):.0f}  p95 {np.percentile(reasoning_out,95):.0f}")
