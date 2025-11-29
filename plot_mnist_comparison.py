"""
Plot comparison charts for MNIST IID vs Non-IID results.
"""
import matplotlib.pyplot as plt
import numpy as np

# Data from results
rounds = [1, 2, 3]

# IID Results
iid_fedavg_acc = [95.99, 97.66, 98.28]
iid_fedavgm_acc = [95.35, 96.57, 97.90]
iid_fedopt_acc = [95.75, 97.26, 98.09]

iid_fedavg_loss = [0.1335, 0.0778, 0.0576]
iid_fedavgm_loss = [0.1497, 0.3406, 0.2351]
iid_fedopt_loss = [0.1538, 0.1247, 0.0866]

# Non-IID Results
noniid_fedavg_acc = [19.43, 35.01, 54.54]
noniid_fedavgm_acc = [22.45, 22.28, 29.45]
noniid_fedopt_acc = [19.80, 31.79, 33.71]

noniid_fedavg_loss = [2.1509, 1.8613, 1.4308]
noniid_fedavgm_loss = [2.1925, 3.6529, 5.6004]
noniid_fedopt_loss = [11.3020, 7.6781, 1.7264]

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('MNIST: Comparison of FedAvg, FedAvgM, FedOpt\n(IID vs Non-IID Distribution)', 
             fontsize=14, fontweight='bold')

# Colors and markers
colors = {'FedAvg': '#2196F3', 'FedAvgM': '#4CAF50', 'FedOpt': '#FF5722'}

# Plot 1: IID Accuracy
ax1 = axes[0, 0]
ax1.plot(rounds, iid_fedavg_acc, marker='o', linewidth=2, markersize=8, 
         color=colors['FedAvg'], label='FedAvg')
ax1.plot(rounds, iid_fedavgm_acc, marker='s', linewidth=2, markersize=8, 
         color=colors['FedAvgM'], label='FedAvgM')
ax1.plot(rounds, iid_fedopt_acc, marker='^', linewidth=2, markersize=8, 
         color=colors['FedOpt'], label='FedOpt')
ax1.set_xlabel('Communication Round', fontsize=11)
ax1.set_ylabel('Test Accuracy (%)', fontsize=11)
ax1.set_title('IID Distribution - Accuracy', fontsize=12, fontweight='bold')
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(rounds)
ax1.set_ylim([94, 99])

# Plot 2: IID Loss
ax2 = axes[0, 1]
ax2.plot(rounds, iid_fedavg_loss, marker='o', linewidth=2, markersize=8, 
         color=colors['FedAvg'], label='FedAvg')
ax2.plot(rounds, iid_fedavgm_loss, marker='s', linewidth=2, markersize=8, 
         color=colors['FedAvgM'], label='FedAvgM')
ax2.plot(rounds, iid_fedopt_loss, marker='^', linewidth=2, markersize=8, 
         color=colors['FedOpt'], label='FedOpt')
ax2.set_xlabel('Communication Round', fontsize=11)
ax2.set_ylabel('Test Loss', fontsize=11)
ax2.set_title('IID Distribution - Loss', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(rounds)

# Plot 3: Non-IID Accuracy
ax3 = axes[1, 0]
ax3.plot(rounds, noniid_fedavg_acc, marker='o', linewidth=2, markersize=8, 
         color=colors['FedAvg'], label='FedAvg')
ax3.plot(rounds, noniid_fedavgm_acc, marker='s', linewidth=2, markersize=8, 
         color=colors['FedAvgM'], label='FedAvgM')
ax3.plot(rounds, noniid_fedopt_acc, marker='^', linewidth=2, markersize=8, 
         color=colors['FedOpt'], label='FedOpt')
ax3.set_xlabel('Communication Round', fontsize=11)
ax3.set_ylabel('Test Accuracy (%)', fontsize=11)
ax3.set_title('Non-IID Distribution - Accuracy', fontsize=12, fontweight='bold')
ax3.legend(loc='upper left', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xticks(rounds)
ax3.set_ylim([15, 60])

# Plot 4: Non-IID Loss
ax4 = axes[1, 1]
ax4.plot(rounds, noniid_fedavg_loss, marker='o', linewidth=2, markersize=8, 
         color=colors['FedAvg'], label='FedAvg')
ax4.plot(rounds, noniid_fedavgm_loss, marker='s', linewidth=2, markersize=8, 
         color=colors['FedAvgM'], label='FedAvgM')
ax4.plot(rounds, noniid_fedopt_loss, marker='^', linewidth=2, markersize=8, 
         color=colors['FedOpt'], label='FedOpt')
ax4.set_xlabel('Communication Round', fontsize=11)
ax4.set_ylabel('Test Loss', fontsize=11)
ax4.set_title('Non-IID Distribution - Loss', fontsize=12, fontweight='bold')
ax4.legend(loc='upper right', fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.set_xticks(rounds)

plt.tight_layout()
plt.savefig('mnist_comparison_chart.png', dpi=150, bbox_inches='tight')
print("Saved: mnist_comparison_chart.png")

# Create bar chart for final results comparison
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle('MNIST: Final Results Comparison (Round 3)', fontsize=14, fontweight='bold')

methods = ['FedAvg', 'FedAvgM', 'FedOpt']
x = np.arange(len(methods))
width = 0.35

# Final Accuracy comparison
ax5 = axes2[0]
iid_final_acc = [98.28, 97.90, 98.09]
noniid_final_acc = [54.54, 29.45, 33.71]
bars1 = ax5.bar(x - width/2, iid_final_acc, width, label='IID', color='#2196F3')
bars2 = ax5.bar(x + width/2, noniid_final_acc, width, label='Non-IID', color='#FF5722')
ax5.set_xlabel('Method', fontsize=11)
ax5.set_ylabel('Final Accuracy (%)', fontsize=11)
ax5.set_title('Final Accuracy Comparison', fontsize=12, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(methods)
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')
ax5.set_ylim([0, 110])
# Add value labels
for bar in bars1:
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)

# Final Loss comparison  
ax6 = axes2[1]
iid_final_loss = [0.0576, 0.2351, 0.0866]
noniid_final_loss = [1.4308, 5.6004, 1.7264]
bars3 = ax6.bar(x - width/2, iid_final_loss, width, label='IID', color='#2196F3')
bars4 = ax6.bar(x + width/2, noniid_final_loss, width, label='Non-IID', color='#FF5722')
ax6.set_xlabel('Method', fontsize=11)
ax6.set_ylabel('Final Loss', fontsize=11)
ax6.set_title('Final Loss Comparison', fontsize=12, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(methods)
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('mnist_final_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: mnist_final_comparison.png")

plt.show()
print("\nDone!")

