"""
Plot comparison charts for MNIST IID vs Non-IID results (including FedNoLowe).
"""
import matplotlib.pyplot as plt
import numpy as np

# Data from results
rounds = [1, 2, 3]

# IID Results
iid_fedavg_acc = [95.99, 97.66, 98.28]
iid_fedavgm_acc = [95.35, 96.57, 97.90]
iid_fedopt_acc = [95.75, 97.26, 98.09]
iid_fednolowe_acc = [95.99, 97.66, 98.28]

iid_fedavg_loss = [0.1335, 0.0778, 0.0576]
iid_fedavgm_loss = [0.1497, 0.3406, 0.2351]
iid_fedopt_loss = [0.1538, 0.1247, 0.0866]
iid_fednolowe_loss = [0.1335, 0.0778, 0.0576]

# Non-IID Results
noniid_fedavg_acc = [19.43, 35.01, 54.54]
noniid_fedavgm_acc = [22.45, 22.28, 29.45]
noniid_fedopt_acc = [19.80, 31.79, 33.71]
noniid_fednolowe_acc = [18.26, 27.57, 55.61]

noniid_fedavg_loss = [2.1509, 1.8613, 1.4308]
noniid_fedavgm_loss = [2.1925, 3.6529, 5.6004]
noniid_fedopt_loss = [11.3020, 7.6781, 1.7264]
noniid_fednolowe_loss = [2.1792, 1.9136, 1.4383]

# Colors
colors = {'FedAvg': '#2196F3', 'FedAvgM': '#4CAF50', 'FedOpt': '#FF5722', 'FedNoLowe': '#9C27B0'}

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('MNIST: Comparison of FedAvg, FedAvgM, FedOpt, FedNoLowe\n(IID vs Non-IID Distribution)', 
             fontsize=14, fontweight='bold')

# Plot 1: IID Accuracy
ax1 = axes[0, 0]
for name, acc, color, marker in [('FedAvg', iid_fedavg_acc, colors['FedAvg'], 'o'),
                                   ('FedAvgM', iid_fedavgm_acc, colors['FedAvgM'], 's'),
                                   ('FedOpt', iid_fedopt_acc, colors['FedOpt'], '^'),
                                   ('FedNoLowe', iid_fednolowe_acc, colors['FedNoLowe'], 'D')]:
    ax1.plot(rounds, acc, marker=marker, linewidth=2, markersize=8, color=color, label=name)
ax1.set_xlabel('Communication Round', fontsize=11)
ax1.set_ylabel('Test Accuracy (%)', fontsize=11)
ax1.set_title('IID Distribution - Accuracy', fontsize=12, fontweight='bold')
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(rounds)
ax1.set_ylim([94, 99])

# Plot 2: IID Loss
ax2 = axes[0, 1]
for name, loss, color, marker in [('FedAvg', iid_fedavg_loss, colors['FedAvg'], 'o'),
                                   ('FedAvgM', iid_fedavgm_loss, colors['FedAvgM'], 's'),
                                   ('FedOpt', iid_fedopt_loss, colors['FedOpt'], '^'),
                                   ('FedNoLowe', iid_fednolowe_loss, colors['FedNoLowe'], 'D')]:
    ax2.plot(rounds, loss, marker=marker, linewidth=2, markersize=8, color=color, label=name)
ax2.set_xlabel('Communication Round', fontsize=11)
ax2.set_ylabel('Test Loss', fontsize=11)
ax2.set_title('IID Distribution - Loss', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(rounds)

# Plot 3: Non-IID Accuracy
ax3 = axes[1, 0]
for name, acc, color, marker in [('FedAvg', noniid_fedavg_acc, colors['FedAvg'], 'o'),
                                   ('FedAvgM', noniid_fedavgm_acc, colors['FedAvgM'], 's'),
                                   ('FedOpt', noniid_fedopt_acc, colors['FedOpt'], '^'),
                                   ('FedNoLowe', noniid_fednolowe_acc, colors['FedNoLowe'], 'D')]:
    ax3.plot(rounds, acc, marker=marker, linewidth=2, markersize=8, color=color, label=name)
ax3.set_xlabel('Communication Round', fontsize=11)
ax3.set_ylabel('Test Accuracy (%)', fontsize=11)
ax3.set_title('Non-IID Distribution - Accuracy', fontsize=12, fontweight='bold')
ax3.legend(loc='upper left', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xticks(rounds)
ax3.set_ylim([15, 60])

# Plot 4: Non-IID Loss
ax4 = axes[1, 1]
for name, loss, color, marker in [('FedAvg', noniid_fedavg_loss, colors['FedAvg'], 'o'),
                                   ('FedAvgM', noniid_fedavgm_loss, colors['FedAvgM'], 's'),
                                   ('FedOpt', noniid_fedopt_loss, colors['FedOpt'], '^'),
                                   ('FedNoLowe', noniid_fednolowe_loss, colors['FedNoLowe'], 'D')]:
    ax4.plot(rounds, loss, marker=marker, linewidth=2, markersize=8, color=color, label=name)
ax4.set_xlabel('Communication Round', fontsize=11)
ax4.set_ylabel('Test Loss', fontsize=11)
ax4.set_title('Non-IID Distribution - Loss', fontsize=12, fontweight='bold')
ax4.legend(loc='upper right', fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.set_xticks(rounds)

plt.tight_layout()
plt.savefig('mnist_comparison_chart_v2.png', dpi=150, bbox_inches='tight')
print("Saved: mnist_comparison_chart_v2.png")

# Bar chart for final results
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('MNIST: Final Results Comparison (Round 3)', fontsize=14, fontweight='bold')

methods = ['FedAvg', 'FedAvgM', 'FedOpt', 'FedNoLowe']
x = np.arange(len(methods))
width = 0.35

# Final Accuracy
ax5 = axes2[0]
iid_final_acc = [98.28, 97.90, 98.09, 98.28]
noniid_final_acc = [54.54, 29.45, 33.71, 55.61]
bars1 = ax5.bar(x - width/2, iid_final_acc, width, label='IID', color='#2196F3', alpha=0.8)
bars2 = ax5.bar(x + width/2, noniid_final_acc, width, label='Non-IID', color='#FF5722', alpha=0.8)
ax5.set_xlabel('Method', fontsize=11)
ax5.set_ylabel('Final Accuracy (%)', fontsize=11)
ax5.set_title('Final Accuracy Comparison', fontsize=12, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(methods)
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')
ax5.set_ylim([0, 110])
for bar in bars1:
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{bar.get_height():.1f}', 
             ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{bar.get_height():.1f}', 
             ha='center', va='bottom', fontsize=9)

# Final Loss
ax6 = axes2[1]
iid_final_loss = [0.0576, 0.2351, 0.0866, 0.0576]
noniid_final_loss = [1.4308, 5.6004, 1.7264, 1.4383]
bars3 = ax6.bar(x - width/2, iid_final_loss, width, label='IID', color='#2196F3', alpha=0.8)
bars4 = ax6.bar(x + width/2, noniid_final_loss, width, label='Non-IID', color='#FF5722', alpha=0.8)
ax6.set_xlabel('Method', fontsize=11)
ax6.set_ylabel('Final Loss', fontsize=11)
ax6.set_title('Final Loss Comparison', fontsize=12, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(methods)
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('mnist_final_comparison_v2.png', dpi=150, bbox_inches='tight')
print("Saved: mnist_final_comparison_v2.png")

plt.show()
print("\nDone!")

