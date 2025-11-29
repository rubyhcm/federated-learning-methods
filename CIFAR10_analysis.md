# Phân Tích Kết Quả Federated Learning trên CIFAR10

## 1. Tổng Quan Thí Nghiệm

### Cấu hình
- **Dataset**: CIFAR10 (50,000 training samples, 10,000 test samples)
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

Trong trường hợp IID, dữ liệu được chia đều cho 10 clients, mỗi client có 5000 samples với đầy đủ 10 classes.

### Bảng kết quả theo rounds

| Round | FedAvg | FedAvgM | FedOpt | FedNoLowe |
|-------|--------|---------|--------|-----------|
| 1     | 35.83% | 35.90%  | 35.58% | 35.83%    |
| 2     | 44.89% | 41.40%  | 40.30% | 44.91%    |
| 3     | 48.97% | 46.99%  | 43.70% | **49.00%**|

### Kết quả cuối cùng (Round 3)

| Method    | Final Acc (%) | Final Loss | Ranking |
|-----------|---------------|------------|---------|
| FedNoLowe | **49.00**     | **1.4173** | 🥇 1    |
| FedAvg    | 48.97         | 1.4175     | 🥈 2    |
| FedAvgM   | 46.99         | 2.8442     | 🥉 3    |
| FedOpt    | 43.70         | 1.6038     | 4       |

### Nhận xét IID
- **FedNoLowe đạt accuracy cao nhất** (49.00%) và loss thấp nhất (1.4173)
- FedAvg và FedNoLowe có hiệu suất gần như tương đương
- FedAvgM có loss cao hơn đáng kể (2.8442) do momentum chưa được tối ưu
- FedOpt có accuracy thấp nhất, có thể do server learning rate chưa phù hợp

---

## 3. Kết Quả Non-IID Distribution

Trong trường hợp Non-IID, mỗi client chỉ có dữ liệu của **1 class duy nhất** (extreme non-IID):
- Client 1: chỉ có class 0
- Client 2: chỉ có class 1
- ...
- Client 10: chỉ có class 9

### Bảng kết quả theo rounds

| Round | FedAvg | FedAvgM | FedOpt | FedNoLowe |
|-------|--------|---------|--------|-----------|
| 1     | 11.89% | 10.02%  | 10.00% | 11.47%    |
| 2     | 13.83% | 9.83%   | 10.00% | **13.90%**|
| 3     | 12.41% | 10.00%  | 10.00% | **12.55%**|

### Kết quả cuối cùng (Round 3)

| Method    | Final Acc (%) | Best Acc (%) | Final Loss | Ranking |
|-----------|---------------|--------------|------------|---------|
| FedNoLowe | **12.55**     | **13.90**    | **2.2918** | 🥇 1    |
| FedAvg    | 12.41         | 13.83        | 2.2918     | 🥈 2    |
| FedAvgM   | 10.00         | 10.02        | 9.5437     | 🥉 3    |
| FedOpt    | 10.00         | 10.00        | 53.9785    | 4       |

### Nhận xét Non-IID
- **FedNoLowe vẫn đạt accuracy cao nhất** (12.55%) trong điều kiện extreme non-IID
- Tất cả phương pháp đều hoạt động kém (~10-13%), gần với random guess (10%)
- FedOpt có loss cực cao (53.9785) - không ổn định với non-IID data
- FedAvgM cũng gặp vấn đề với loss tăng cao (9.5437)

---

## 4. So Sánh IID vs Non-IID

| Method    | IID Acc | Non-IID Acc | Độ giảm    |
|-----------|---------|-------------|------------|
| FedNoLowe | 49.00%  | 12.55%      | -36.45%    |
| FedAvg    | 48.97%  | 12.41%      | -36.56%    |
| FedAvgM   | 46.99%  | 10.00%      | -36.99%    |
| FedOpt    | 43.70%  | 10.00%      | -33.70%    |

---

## 5. Kết Luận

### 🏆 Phương pháp tốt nhất: **FedNoLowe**
- Đạt accuracy cao nhất trong cả hai trường hợp IID và Non-IID
- Loss ổn định và thấp nhất
- Cải thiện nhẹ so với FedAvg gốc

### 📊 Xếp hạng tổng thể
1. **FedNoLowe** - Tốt nhất, ổn định
2. **FedAvg** - Gần tương đương FedNoLowe
3. **FedAvgM** - Cần điều chỉnh hyperparameters
4. **FedOpt** - Không ổn định, cần tuning

### ⚠️ Thách thức với Non-IID
- Extreme non-IID (mỗi client 1 class) là thách thức lớn nhất
- Accuracy giảm ~36% so với IID
- Cần nhiều communication rounds hơn hoặc các kỹ thuật đặc biệt (FedProx, SCAFFOLD, ...)

### 💡 Khuyến nghị
1. Tăng số rounds lên 50-100 để thấy sự khác biệt rõ hơn
2. Thử nghiệm với mild non-IID (mỗi client 2-3 classes)
3. Điều chỉnh server_momentum và server_lr cho FedAvgM và FedOpt
4. Xem xét các phương pháp xử lý non-IID như FedProx, SCAFFOLD

---

## 6. Biểu Đồ

Xem các file:
- `cifar10_comparison_chart_v2.png` - Biểu đồ accuracy và loss theo rounds
- `cifar10_final_comparison_v2.png` - Biểu đồ so sánh kết quả cuối cùng

