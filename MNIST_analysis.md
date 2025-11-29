# Phân Tích Kết Quả Federated Learning trên MNIST

## 1. Tổng Quan Thí Nghiệm

### Cấu hình
- **Dataset**: MNIST (60,000 training samples, 10,000 test samples)
- **Số clients**: 10
- **Communication rounds**: 3
- **Local epochs**: 5
- **Batch size**: 32
- **Learning rate**: 0.01

### Phương pháp so sánh
1. **FedAvg**: Federated Averaging cơ bản
2. **FedAvgM**: FedAvg với Server Momentum
3. **FedOpt**: Federated Optimization với Adam optimizer
4. **FedNoLowe**: Phương pháp mới được đề xuất

---

## 2. Kết Quả IID Distribution

Trong trường hợp IID, dữ liệu được chia đều cho 10 clients, mỗi client có 6000 samples với đầy đủ 10 classes.

### Bảng kết quả theo rounds

| Round | FedAvg | FedAvgM | FedOpt | FedNoLowe |
|-------|--------|---------|--------|-----------|
| 1     | 95.99% | 95.35%  | 95.75% | 95.99%    |
| 2     | 97.66% | 96.57%  | 97.26% | 97.66%    |
| 3     | 98.28% | 97.90%  | 98.09% | **98.28%**|

### Kết quả cuối cùng (Round 3)

| Method    | Final Acc (%) | Final Loss | Ranking |
|-----------|---------------|------------|---------|
| FedAvg    | **98.28**     | **0.0576** | 🥇 1 (tie) |
| FedNoLowe | **98.28**     | **0.0576** | 🥇 1 (tie) |
| FedOpt    | 98.09         | 0.0866     | 🥉 3    |
| FedAvgM   | 97.90         | 0.2351     | 4       |

### Nhận xét IID
- **FedAvg và FedNoLowe đạt kết quả giống hệt nhau** (98.28% accuracy, 0.0576 loss)
- MNIST là dataset đơn giản, tất cả phương pháp đều đạt >97% accuracy
- FedAvgM có loss cao hơn (0.2351) do momentum gây dao động
- FedOpt hoạt động tốt với accuracy 98.09%

---

## 3. Kết Quả Non-IID Distribution

Trong trường hợp Non-IID, mỗi client chỉ có dữ liệu của **1-2 classes**:
- Client 1: classes [0, 1]
- Client 2: class [1]
- Client 3: classes [1, 2]
- ...

### Bảng kết quả theo rounds

| Round | FedAvg | FedAvgM | FedOpt | FedNoLowe |
|-------|--------|---------|--------|-----------|
| 1     | 19.43% | 22.45%  | 19.80% | 18.26%    |
| 2     | 35.01% | 22.28%  | 31.79% | 27.57%    |
| 3     | 54.54% | 29.45%  | 33.71% | **55.61%**|

### Kết quả cuối cùng (Round 3)

| Method    | Final Acc (%) | Final Loss | Ranking |
|-----------|---------------|------------|---------|
| FedNoLowe | **55.61**     | 1.4383     | 🥇 1    |
| FedAvg    | 54.54         | **1.4308** | 🥈 2    |
| FedOpt    | 33.71         | 1.7264     | 🥉 3    |
| FedAvgM   | 29.45         | 5.6004     | 4       |

### Nhận xét Non-IID
- **FedNoLowe đạt accuracy cao nhất** (55.61%) - vượt FedAvg 1.07%
- FedNoLowe có tốc độ hội tụ chậm hơn ở round 1-2 nhưng bứt phá ở round 3
- FedAvgM và FedOpt hoạt động kém với non-IID data
- FedAvgM có loss rất cao (5.6004) - không ổn định

---

## 4. So Sánh IID vs Non-IID

| Method    | IID Acc | Non-IID Acc | Độ giảm    |
|-----------|---------|-------------|------------|
| FedNoLowe | 98.28%  | 55.61%      | -42.67%    |
| FedAvg    | 98.28%  | 54.54%      | -43.74%    |
| FedOpt    | 98.09%  | 33.71%      | -64.38%    |
| FedAvgM   | 97.90%  | 29.45%      | -68.45%    |

### Điểm nổi bật
- FedNoLowe giữ được hiệu suất tốt nhất trong điều kiện Non-IID
- Độ giảm accuracy của FedNoLowe (-42.67%) thấp hơn FedAvg (-43.74%)
- FedAvgM và FedOpt bị ảnh hưởng nặng bởi non-IID data

---

## 5. So Sánh với CIFAR10

| Dataset | Method    | IID Acc | Non-IID Acc |
|---------|-----------|---------|-------------|
| MNIST   | FedNoLowe | 98.28%  | 55.61%      |
| MNIST   | FedAvg    | 98.28%  | 54.54%      |
| CIFAR10 | FedNoLowe | 49.00%  | 12.55%      |
| CIFAR10 | FedAvg    | 48.97%  | 12.41%      |

- MNIST dễ hơn CIFAR10 đáng kể (98% vs 49%)
- Non-IID ảnh hưởng nghiêm trọng hơn trên CIFAR10

---

## 6. Kết Luận

### 🏆 Phương pháp tốt nhất: **FedNoLowe**
- IID: Tương đương FedAvg (98.28%)
- Non-IID: **Tốt nhất** (55.61%, cao hơn FedAvg 1.07%)
- Loss ổn định trong cả hai trường hợp

### 📊 Xếp hạng tổng thể
1. **FedNoLowe** - Tốt nhất cho Non-IID
2. **FedAvg** - Ổn định, baseline tốt
3. **FedOpt** - Cần tuning hyperparameters
4. **FedAvgM** - Không phù hợp với ít rounds

### 💡 Khuyến nghị
1. Sử dụng **FedNoLowe** khi có non-IID data
2. Tăng số rounds để thấy sự khác biệt rõ hơn
3. Điều chỉnh momentum và learning rate cho FedAvgM/FedOpt

---

## 7. Biểu Đồ

Xem các file:
- `mnist_comparison_chart_v2.png` - Biểu đồ accuracy và loss theo rounds
- `mnist_final_comparison_v2.png` - Biểu đồ so sánh kết quả cuối cùng

