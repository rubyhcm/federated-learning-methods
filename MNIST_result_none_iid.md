<!-- cd /home/loinguyen/federated-learning-methods && source venv/bin/activate && python train_federated.py --dataset mnist --distribution non-iid --num-rounds 3 --no-plot -->

======================================================================
FEDERATED LEARNING TRAINING
======================================================================
Dataset: MNIST
Distribution: NON-IID
Device: cpu
Number of clients: 10
Communication rounds: 3
Local epochs: 5
Batch size: 32
Learning rate: 0.01
======================================================================

Loading MNIST dataset...
Creating federated data (Non-IID distribution)...
Total training samples: 60000
Total test samples: 10000
Samples per client: ~6000

======================================================================
Data Distribution Across Clients
======================================================================
Client 1: 6000 samples, Classes: [0, 1]
Client 2: 6000 samples, Classes: [1]
Client 3: 6000 samples, Classes: [1, 2]
Client 4: 6000 samples, Classes: [2, 3]
Client 5: 6000 samples, Classes: [3, 4]
Client 6: 6000 samples, Classes: [4, 5]
Client 7: 6000 samples, Classes: [5, 6, 7]
Client 8: 6000 samples, Classes: [7]
Client 9: 6000 samples, Classes: [7, 8]
Client 10: 6000 samples, Classes: [8, 9]
======================================================================

======================================================================
Training with FedAvg
======================================================================
Round 1/3 - Test Accuracy: 19.43%, Test Loss: 2.1509
Round 2/3 - Test Accuracy: 35.01%, Test Loss: 1.8613
Round 3/3 - Test Accuracy: 54.54%, Test Loss: 1.4308

======================================================================
Training with FedAvgM
======================================================================
Round 1/3 - Test Accuracy: 22.45%, Test Loss: 2.1925
Round 2/3 - Test Accuracy: 22.28%, Test Loss: 3.6529
Round 3/3 - Test Accuracy: 29.45%, Test Loss: 5.6004

======================================================================
Training with FedOpt
======================================================================
Round 1/3 - Test Accuracy: 19.80%, Test Loss: 11.3020
Round 2/3 - Test Accuracy: 31.79%, Test Loss: 7.6781
Round 3/3 - Test Accuracy: 33.71%, Test Loss: 1.7264

======================================================================
Training with FedNoLowe
======================================================================
Round 1/3 - Test Accuracy: 18.26%, Test Loss: 2.1792
Round 2/3 - Test Accuracy: 27.57%, Test Loss: 1.9136
Round 3/3 - Test Accuracy: 55.61%, Test Loss: 1.4383

======================================================================
FINAL RESULTS - MNIST - NON-IID
======================================================================
Method          Final Acc (%)   Best Acc (%)    Final Loss     
----------------------------------------------------------------------
FedAvg          54.54           54.54           1.4308         
FedAvgM         29.45           29.45           5.6004         
FedOpt          33.71           33.71           1.7264        
FedNoLowe       55.61           55.61           1.4383   
======================================================================
