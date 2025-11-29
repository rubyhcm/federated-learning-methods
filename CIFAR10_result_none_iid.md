======================================================================
FEDERATED LEARNING TRAINING
======================================================================
Dataset: CIFAR10
Distribution: NON-IID
Device: cpu
Number of clients: 10
Communication rounds: 3
Local epochs: 5
Batch size: 32
Learning rate: 0.01
======================================================================

Loading CIFAR10 dataset...
100.0%
Creating federated data (Non-IID distribution)...
Total training samples: 50000
Total test samples: 10000
Samples per client: ~5000

======================================================================
Data Distribution Across Clients
======================================================================
Client 1: 5000 samples, Classes: [0]
Client 2: 5000 samples, Classes: [1]
Client 3: 5000 samples, Classes: [2]
Client 4: 5000 samples, Classes: [3]
Client 5: 5000 samples, Classes: [4]
Client 6: 5000 samples, Classes: [5]
Client 7: 5000 samples, Classes: [6]
Client 8: 5000 samples, Classes: [7]
Client 9: 5000 samples, Classes: [8]
Client 10: 5000 samples, Classes: [9]
======================================================================


======================================================================
Training with FedAvg
======================================================================
Round 1/3 - Test Accuracy: 11.89%, Test Loss: 2.3247
Round 2/3 - Test Accuracy: 13.83%, Test Loss: 2.3144
Round 3/3 - Test Accuracy: 12.41%, Test Loss: 2.2918

======================================================================
Training with FedAvgM
======================================================================
Round 1/3 - Test Accuracy: 10.02%, Test Loss: 2.3389
Round 2/3 - Test Accuracy: 9.83%, Test Loss: 2.4303
Round 3/3 - Test Accuracy: 10.00%, Test Loss: 9.5437

======================================================================
Training with FedOpt
======================================================================
Round 1/3 - Test Accuracy: 10.00%, Test Loss: 12.3512
Round 2/3 - Test Accuracy: 10.00%, Test Loss: 119.0757
Round 3/3 - Test Accuracy: 10.00%, Test Loss: 53.9785

Training with FedNoLowe
======================================================================
Round 1/3 - Test Accuracy: 11.47%, Test Loss: 2.3239
Round 2/3 - Test Accuracy: 13.90%, Test Loss: 2.3141
Round 3/3 - Test Accuracy: 12.55%, Test Loss: 2.2918

======================================================================
FINAL RESULTS - CIFAR10 - NON-IID
======================================================================
Method          Final Acc (%)   Best Acc (%)    Final Loss     
----------------------------------------------------------------------
FedAvg          12.41           13.83           2.2918         
FedAvgM         10.00           10.02           9.5437         
FedOpt          10.00           10.00           53.9785   
FedNoLowe       12.55           13.90           2.2918        
======================================================================
