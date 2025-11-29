<!-- cd /home/loinguyen/federated-learning-methods && source venv/bin/activate && python train_federated.py --dataset mnist --distribution iid --num-rounds 3 --no-plot -->

======================================================================
FEDERATED LEARNING TRAINING
======================================================================
Dataset: MNIST
Distribution: IID
Device: cpu
Number of clients: 10
Communication rounds: 3
Local epochs: 5
Batch size: 32
Learning rate: 0.01
======================================================================

Loading MNIST dataset...
Creating federated data (IID distribution)...
Total training samples: 60000
Total test samples: 10000
Samples per client: ~6000

======================================================================
Data Distribution Across Clients
======================================================================
Client 1: 6000 samples, Classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Client 2: 6000 samples, Classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Client 3: 6000 samples, Classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Client 4: 6000 samples, Classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Client 5: 6000 samples, Classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Client 6: 6000 samples, Classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Client 7: 6000 samples, Classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Client 8: 6000 samples, Classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Client 9: 6000 samples, Classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Client 10: 6000 samples, Classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
======================================================================


======================================================================
Training with FedAvg
======================================================================
Round 1/3 - Test Accuracy: 95.99%, Test Loss: 0.1335
Round 2/3 - Test Accuracy: 97.66%, Test Loss: 0.0778
Round 3/3 - Test Accuracy: 98.28%, Test Loss: 0.0576

======================================================================
Training with FedAvgM
======================================================================
Round 1/3 - Test Accuracy: 95.35%, Test Loss: 0.1497
Round 2/3 - Test Accuracy: 96.57%, Test Loss: 0.3406
Round 3/3 - Test Accuracy: 97.90%, Test Loss: 0.2351

======================================================================
Training with FedOpt
======================================================================
Round 1/3 - Test Accuracy: 95.75%, Test Loss: 0.1538
Round 2/3 - Test Accuracy: 97.26%, Test Loss: 0.1247
Round 3/3 - Test Accuracy: 98.09%, Test Loss: 0.0866

======================================================================
Training with FedNoLowe
======================================================================
Round 1/3 - Test Accuracy: 95.99%, Test Loss: 0.1335
Round 2/3 - Test Accuracy: 97.66%, Test Loss: 0.0778
Round 3/3 - Test Accuracy: 98.28%, Test Loss: 0.0576

======================================================================
FINAL RESULTS - MNIST - IID
======================================================================
Method          Final Acc (%)   Best Acc (%)    Final Loss     
----------------------------------------------------------------------
FedAvg          98.28           98.28           0.0576         
FedAvgM         97.90           97.90           0.2351         
FedOpt          98.09           98.09           0.0866
FedNoLowe       98.28           98.28           0.0576         
======================================================================
