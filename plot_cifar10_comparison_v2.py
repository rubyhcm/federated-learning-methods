"""
Plot comparison charts for CIFAR10 IID vs Non-IID results (including FedNoLowe).
"""
import matplotlib.pyplot as plt
import numpy as np

# Data from results
rounds = [1, 2, 3]

# IID Results
iid_fedavg_acc = [35.83, 44.89, 48.97]
iid_fedavgm_acc = [35.90, 41.40, 46.99]
iid_fedopt_acc = [35.58, 40.30, 43.70]
iid_fednolowe_acc = [35.83, 44.91, 49.00]

iid_fedavg_loss = [1.7902, 1.5288, 1.4175]
iid_fedavgm_loss = [1.7900, 2.8750, 2.8442]
iid_fedopt_loss = [4.0572, 2.1810, 1.6038]
iid_fednolowe_loss = [1.7902, 1.5290, 1.4173]

# Non-IID Results
noniid_fedavg_acc = [11.89, 13.83, 12.41]
noniid_fedavgm_acc = [10.02, 9.83, 10.00]
noniid_fedopt_acc = [10.00, 10.00, 10.00]
noniid_fednolowe_acc = [11.47, 13.90, 12.55]

noniid_fedavg_loss = [2.3247, 2.3144, 2.2918]
noniid_fedavgm_loss = [2.3389, 2.4303, 9.5437]
noniid_fedopt_loss = [12.3512, 119.0757, 53.9785]
noniid_fednolowe_loss = [2.3239, 2.3141, 2.2918]

# Colors
colors = {'FedAvg': '#2196F3', 'FedAvgM': '#4CAF50', 'FedOpt': '#FF5722', 'FedNoLowe': '#9C27B0'}
markers = {'FedAvg': 'o', 'FedAvgM': 's', 'FedOpt': '^', 'FedNoLowe': 'D'}

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('CIFAR10: Comparison of FedAvg, FedAvgM, FedOpt, FedNoLowe\n(IID vs Non-IID Distribution)', 
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
ax1.set_ylim([30, 55])

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
ax3.legend(loc='upper right', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xticks(rounds)
ax3.set_ylim([8, 16])

# Plot 4: Non-IID Loss (log scale)
ax4 = axes[1, 1]
for name, loss, color, marker in [('FedAvg', noniid_fedavg_loss, colors['FedAvg'], 'o'),
                                   ('FedAvgM', noniid_fedavgm_loss, colors['FedAvgM'], 's'),
                                   ('FedOpt', noniid_fedopt_loss, colors['FedOpt'], '^'),
                                   ('FedNoLowe', noniid_fednolowe_loss, colors['FedNoLowe'], 'D')]:
    ax4.plot(rounds, loss, marker=marker, linewidth=2, markersize=8, color=color, label=name)
ax4.set_xlabel('Communication Round', fontsize=11)
ax4.set_ylabel('Test Loss (log scale)', fontsize=11)
ax4.set_title('Non-IID Distribution - Loss', fontsize=12, fontweight='bold')
ax4.legend(loc='upper right', fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.set_xticks(rounds)
ax4.set_yscale('log')

plt.tight_layout()
plt.savefig('cifar10_comparison_chart_v2.png', dpi=150, bbox_inches='tight')
print("Saved: cifar10_comparison_chart_v2.png")

# Bar chart for final results
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('CIFAR10: Final Results Comparison (Round 3)', fontsize=14, fontweight='bold')

methods = ['FedAvg', 'FedAvgM', 'FedOpt', 'FedNoLowe']
x = np.arange(len(methods))
width = 0.35
method_colors = [colors[m] for m in methods]

# Final Accuracy
ax5 = axes2[0]
iid_final_acc = [48.97, 46.99, 43.70, 49.00]
noniid_final_acc = [12.41, 10.00, 10.00, 12.55]
bars1 = ax5.bar(x - width/2, iid_final_acc, width, label='IID', color='#2196F3', alpha=0.8)
bars2 = ax5.bar(x + width/2, noniid_final_acc, width, label='Non-IID', color='#FF5722', alpha=0.8)
ax5.set_xlabel('Method', fontsize=11)
ax5.set_ylabel('Final Accuracy (%)', fontsize=11)
ax5.set_title('Final Accuracy Comparison', fontsize=12, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(methods)
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')
for bar in bars1:
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{bar.get_height():.1f}', 
             ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{bar.get_height():.1f}', 
             ha='center', va='bottom', fontsize=9)

# Final Loss
ax6 = axes2[1]
iid_final_loss = [1.4175, 2.8442, 1.6038, 1.4173]
noniid_final_loss = [2.2918, 9.5437, 53.9785, 2.2918]
bars3 = ax6.bar(x - width/2, iid_final_loss, width, label='IID', color='#2196F3', alpha=0.8)
bars4 = ax6.bar(x + width/2, noniid_final_loss, width, label='Non-IID', color='#FF5722', alpha=0.8)
ax6.set_xlabel('Method', fontsize=11)
ax6.set_ylabel('Final Loss (log scale)', fontsize=11)
ax6.set_title('Final Loss Comparison', fontsize=12, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(methods)
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')
ax6.set_yscale('log')

plt.tight_layout()
plt.savefig('cifar10_final_comparison_v2.png', dpi=150, bbox_inches='tight')
print("Saved: cifar10_final_comparison_v2.png")

plt.show()
print("\nDone!")

