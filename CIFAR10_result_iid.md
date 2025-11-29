======================================================================
FEDERATED LEARNING TRAINING
======================================================================
Dataset: CIFAR10
Distribution: IID
Device: cpu
Number of clients: 10
Communication rounds: 3
Local epochs: 5
Batch size: 32
Learning rate: 0.01
======================================================================

Loading CIFAR10 dataset...
100.0%
Creating federated data (IID distribution)...
Total training samples: 50000
Total test samples: 10000
Samples per client: ~5000

======================================================================
Data Distribution Across Clients
======================================================================
Client 1: 5000 samples, Classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Client 2: 5000 samples, Classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Client 3: 5000 samples, Classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Client 4: 5000 samples, Classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Client 5: 5000 samples, Classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Client 6: 5000 samples, Classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Client 7: 5000 samples, Classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Client 8: 5000 samples, Classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Client 9: 5000 samples, Classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Client 10: 5000 samples, Classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
======================================================================


======================================================================
Training with FedAvg
======================================================================
Round 1/3 - Test Accuracy: 35.83%, Test Loss: 1.7902
Round 2/3 - Test Accuracy: 44.89%, Test Loss: 1.5288
Round 3/3 - Test Accuracy: 48.97%, Test Loss: 1.4175

======================================================================
Training with FedAvgM
======================================================================
Round 1/3 - Test Accuracy: 35.90%, Test Loss: 1.7900
Round 2/3 - Test Accuracy: 41.40%, Test Loss: 2.8750
Round 3/3 - Test Accuracy: 46.99%, Test Loss: 2.8442

======================================================================
Training with FedOpt
======================================================================
Round 1/3 - Test Accuracy: 35.58%, Test Loss: 4.0572
Round 2/3 - Test Accuracy: 40.30%, Test Loss: 2.1810
Round 3/3 - Test Accuracy: 43.70%, Test Loss: 1.6038

======================================================================
Training with FedNoLowe
======================================================================
Round 1/3 - Test Accuracy: 35.83%, Test Loss: 1.7902
Round 2/3 - Test Accuracy: 44.91%, Test Loss: 1.5290
Round 3/3 - Test Accuracy: 49.00%, Test Loss: 1.4173

======================================================================
FINAL RESULTS - CIFAR10 - IID
======================================================================
Method          Final Acc (%)   Best Acc (%)    Final Loss     
----------------------------------------------------------------------
FedAvg          48.97           48.97           1.4175         
FedAvgM         46.99           46.99           2.8442         
FedOpt          43.70           43.70           1.6038
FedNoLowe       49.00           49.00           1.4173             
======================================================================
