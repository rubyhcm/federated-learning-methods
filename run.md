# Hướng Dẫn Chạy Train Federated Learning

## 1. Cài Đặt Môi Trường

```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt môi trường
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows

# Cài đặt dependencies
pip install torch torchvision matplotlib numpy
```

---

## 2. Cú Pháp Lệnh Chung

```bash
python train_federated.py [OPTIONS]
```

### Các tham số chính:

| Tham số | Mô tả | Giá trị mặc định |
|---------|-------|------------------|
| `--dataset` | Dataset sử dụng | `mnist` hoặc `cifar10` |
| `--distribution` | Cách chia dữ liệu | `iid` hoặc `non-iid` |
| `--method` | Phương pháp FL | `all`, `fedavg`, `fedavgm`, `fedopt`, `fednolowe` |
| `--num-rounds` | Số vòng communication | 50 |
| `--num-clients` | Số clients | 10 |
| `--local-epochs` | Số epochs local training | 5 |
| `--batch-size` | Batch size | 32 |
| `--learning-rate` | Learning rate | 0.01 |
| `--server-momentum` | Momentum cho FedAvgM | 0.9 |
| `--server-lr` | Server learning rate cho FedOpt | 0.01 |
| `--server-optimizer` | Optimizer cho FedOpt | `adam` |
| `--no-plot` | Không hiển thị biểu đồ | - |

---

## 3. Chạy Từng Phương Pháp

### 3.1 FedAvg (Federated Averaging)

```bash
# MNIST - IID
python train_federated.py --dataset mnist --distribution iid --method fedavg --num-rounds 50

# MNIST - Non-IID
python train_federated.py --dataset mnist --distribution non-iid --method fedavg --num-rounds 50

# CIFAR10 - IID
python train_federated.py --dataset cifar10 --distribution iid --method fedavg --num-rounds 50

# CIFAR10 - Non-IID
python train_federated.py --dataset cifar10 --distribution non-iid --method fedavg --num-rounds 50
```

### 3.2 FedAvgM (FedAvg với Server Momentum)

```bash
# MNIST - IID
python train_federated.py --dataset mnist --distribution iid --method fedavgm --server-momentum 0.9

# MNIST - Non-IID
python train_federated.py --dataset mnist --distribution non-iid --method fedavgm --server-momentum 0.9

# CIFAR10 - IID
python train_federated.py --dataset cifar10 --distribution iid --method fedavgm --server-momentum 0.9

# CIFAR10 - Non-IID
python train_federated.py --dataset cifar10 --distribution non-iid --method fedavgm --server-momentum 0.9
```

### 3.3 FedOpt (Federated Optimization)

```bash
# MNIST - IID (với Adam optimizer)
python train_federated.py --dataset mnist --distribution iid --method fedopt --server-optimizer adam --server-lr 0.01

# MNIST - Non-IID
python train_federated.py --dataset mnist --distribution non-iid --method fedopt --server-optimizer adam --server-lr 0.01

# CIFAR10 - IID
python train_federated.py --dataset cifar10 --distribution iid --method fedopt --server-optimizer adam --server-lr 0.01

# CIFAR10 - Non-IID
python train_federated.py --dataset cifar10 --distribution non-iid --method fedopt --server-optimizer adam --server-lr 0.01
```

### 3.4 FedNoLowe

```bash
# MNIST - IID
python train_federated.py --dataset mnist --distribution iid --method fednolowe --num-rounds 50

# MNIST - Non-IID
python train_federated.py --dataset mnist --distribution non-iid --method fednolowe --num-rounds 50

# CIFAR10 - IID
python train_federated.py --dataset cifar10 --distribution iid --method fednolowe --num-rounds 50

# CIFAR10 - Non-IID
python train_federated.py --dataset cifar10 --distribution non-iid --method fednolowe --num-rounds 50
```

---

## 4. Chạy Tất Cả Phương Pháp Cùng Lúc

```bash
# MNIST - IID - Tất cả methods
python train_federated.py --dataset mnist --distribution iid --method all --num-rounds 50

# MNIST - Non-IID - Tất cả methods
python train_federated.py --dataset mnist --distribution non-iid --method all --num-rounds 50

# CIFAR10 - IID - Tất cả methods
python train_federated.py --dataset cifar10 --distribution iid --method all --num-rounds 50

# CIFAR10 - Non-IID - Tất cả methods
python train_federated.py --dataset cifar10 --distribution non-iid --method all --num-rounds 50
```

---

## 5. Ví Dụ Quick Test (3 rounds)

```bash
# Test nhanh MNIST
python train_federated.py --dataset mnist --distribution iid --method all --num-rounds 3 --no-plot

# Test nhanh CIFAR10
python train_federated.py --dataset cifar10 --distribution iid --method all --num-rounds 3 --no-plot
```

---

## 6. Vẽ Biểu Đồ So Sánh

```bash
# Vẽ biểu đồ CIFAR10
python plot_cifar10_comparison_v2.py

# Vẽ biểu đồ MNIST
python plot_mnist_comparison_v2.py
```

---

## 7. Kết Quả Đầu Ra

- **Console**: Hiển thị accuracy và loss sau mỗi round
- **Biểu đồ**: Tự động hiển thị sau khi train xong (nếu không dùng `--no-plot`)
- **Files kết quả**: Lưu vào các file `.md` để phân tích

---

## 8. Lưu Ý

1. **IID Distribution**: Dữ liệu chia đều, mỗi client có tất cả classes
2. **Non-IID Distribution**: Mỗi client chỉ có 1-2 classes (dữ liệu mất cân bằng)
3. **Số rounds**: Nên chạy ít nhất 50 rounds để thấy sự khác biệt rõ ràng
4. **FedOpt**: Cần điều chỉnh `--server-lr` phù hợp (0.01 hoặc 0.001)
5. **FedAvgM**: Thử các giá trị `--server-momentum` từ 0.5 đến 0.99

