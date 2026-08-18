# Tiến độ dự án FedLLM Influence Tracing

**Cập nhật:** 2026-08-18  
**Nhánh:** `main`  
**Mốc code LLM gần nhất:** `ef51b2a` — `Add CPU-tested LoRA and ProToken pipeline`

## 1. Mục tiêu nghiên cứu

Dự án nghiên cứu khả năng truy vết client nào đã đóng góp vào một hành vi sinh cụ
thể của Federated LLM khi server không quan sát được update cá nhân do Secure
Aggregation.

Pipeline mục tiêu:

```text
Secure aggregate queries
→ FedAttr client-update estimate
→ estimated client LoRA model
→ token-level ProToken attribution
→ multi-round historical provenance
```

Câu hỏi khả thi trung tâm là liệu update nhiễu
`estimated_delta = true_delta + noise` có giữ đủ thông tin để bảo toàn attribution
score và thứ hạng client hay không.

Điểm số hiện được xem là **historical behavioral provenance**, không phải causal
influence chính xác. Dự án chưa triển khai federated unlearning hoặc
counterfactual retraining.

## 2. Trạng thái hiện tại

### 2.1. Hạ tầng federated và FedAttr — hoàn thành

- `ClientUpdate` lưu LoRA-only state dict cùng client ID, round ID và số mẫu.
- Tensor-state operations không sửa input và kiểm tra key/shape/dtype/device.
- FedAvg hỗ trợ weighted và unweighted aggregation.
- Mock federated simulator sinh update tái lập được theo seed.
- Secure Aggregation oracle chỉ trả subset sums.
- FedAttr estimator dùng các cặp target-present/target-absent được lấy mẫu đối
  xứng.
- Controlled relative-L2 noise hỗ trợ các thí nghiệm sensitivity.
- Metrics đã có: attribution MAE, Spearman, Top-1, Top-k overlap và NDCG.
- Safetensors serialization, config hashing và JSONL logging đã hoạt động.

### 2.2. LLM, LoRA và ProToken — hoàn thành ở mức CPU validation

- Đã cài `transformers`, `peft`, `datasets` và `accelerate` trong `.venv`.
- Model loader hỗ trợ lựa chọn CPU/CUDA và dtype rõ ràng.
- PEFT utilities hỗ trợ attach, extract, load và tạm swap LoRA adapter.
- Mọi federated client phải dùng cùng base weights và cùng initial LoRA adapter
  trước local training.
- Teacher-forcing masking và padding đã được triển khai.
- Local training loop chỉ tối ưu các parameter đang được đánh dấu trainable.
- `PeftTokenTracer` đã hiện thực token-level attribution cho PEFT models.

ProToken implementation tuân thủ invariant quan trọng:

1. Global model thực hiện một full forward pass.
2. Tracer capture chính xác positional args và keyword args của từng global
   layer call, gồm attention mask, position IDs và position embeddings.
3. Target-token logit gradient được tính từ global computation graph.
4. Với mỗi client, chỉ selected layers được replay bằng đúng global layer input
   đã capture sau khi swap client LoRA update.
5. Không thực hiện một client-specific full-model forward pass độc lập.

### 2.3. Multi-round — mới có baseline interface

Đã hỗ trợ các aggregation mode `sum`, `mean`, `last` và `weighted_mean`. Chưa có
multi-round FL experiment và chưa chọn mode nào làm phương pháp nghiên cứu cuối
cùng.

## 3. Những gì chạy được ngay

Tạo/cập nhật môi trường:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev,llm]"
```

Chạy toàn bộ test:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Chạy FedAttr mock experiment:

```powershell
.\.venv\Scripts\python.exe scripts\test_fedattr_mock.py
```

Chạy LoRA–ProToken end-to-end CPU smoke:

```powershell
.\.venv\Scripts\python.exe scripts\run_protoken_cpu_smoke.py
```

Kiểm tra truy cập config/tokenizer Qwen thật mà không tải model weights:

```powershell
.\.venv\Scripts\python.exe scripts\check_model_access.py `
  --model Qwen/Qwen2.5-0.5B
```

Mỗi experiment tạo một thư mục riêng dưới `outputs/`. Run artifacts không được
commit vào Git.

## 4. Kết quả kiểm chứng hiện có

### 4.1. Automated tests

```text
44 passed
```

Tests bao phủ FedAvg, subset sums, paired sampling, FedAttr convergence,
controlled noise, metrics, serialization, result logging, LoRA state handling,
local adapter training và exact-global-input layer replay.

### 4.2. FedAttr mock

Thiết lập: 10 clients, update dimension 10.000, subset size 4.

| Paired queries M | Mean relative update error |
|---:|---:|
| 2 | 1.481676 |
| 3 | 1.118279 |
| 5 | 0.922652 |
| 10 | 0.688686 |
| 20 | 0.452812 |

Kết quả xác nhận estimation error giảm khi tăng số query pairs. Đây chỉ là
implementation sanity check trên random tensors.

### 4.3. Tiny-Qwen ProToken smoke

Thiết lập: random tiny Qwen2, 3 clients, 2 decoder layers, LoRA rank 2, một token
fact riêng cho mỗi client và 6 local steps.

- Local training loss giảm cho cả ba client.
- Oracle ProToken xếp đúng client chứa target token ở Top-1 cho cả ba prompts.
- Noise `0.0` cho MAE `0`, Spearman `1`, Top-1 `true`, NDCG `1`.
- Khi noise tăng lên `0.5`, MAE tăng; một prompt giảm Spearman xuống `0.5` nhưng
  Top-1 vẫn đúng.
- Run gần nhất ghi 27 JSONL records: 3 prompts × 3 noise levels × 3 clients.

Model dùng random weights và token IDs nhân tạo, vì vậy kết quả này **không phải
kết quả khoa học** và không được dùng để đưa ra kết luận về Qwen pretrained.

### 4.4. Qwen2.5-0.5B preflight

Config/tokenizer access đã thành công mà không tải model weights:

```text
architecture: Qwen2ForCausalLM
layers:       24
hidden size:  896
vocab size:   151936
tokenizer:    Qwen2Tokenizer
```

Máy hiện dùng PyTorch CPU và `torch.cuda.is_available()` trả về `False`.

## 5. Phần còn thiếu hoặc đang chờ

### Chờ môi trường GPU

- Cài PyTorch build tương ứng với CUDA của máy GPU.
- Tải Qwen2.5-0.5B pretrained weights.
- Xác minh memory usage cho forward, backward và selected-layer replay.
- Chạy local LoRA training với dữ liệu văn bản thật.

### Cần bổ sung cho kết quả sơ bộ

- Khóa controlled text dataset và prompt/target set cho ba client.
- Thêm config-driven pretrained Qwen experiment entrypoint.
- Lưu initial adapter, client updates và FedAvg adapter thành checkpoints.
- Chạy oracle ProToken với 5–10 tracing prompts.
- Chạy đủ noise levels `0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0`.
- Lặp ít nhất ba seeds và tổng hợp mean/std hoặc confidence intervals.
- Xuất summary CSV và biểu đồ noise → MAE/Spearman/Top-1/NDCG.
- Sau khi oracle/noise pipeline ổn định, thay controlled noise bằng FedAttr
  estimates và đánh giá theo số query pairs `M`.

### Chưa làm trong giai đoạn này

- Multi-round training/tracing experiment.
- Partial participation và non-IID stress tests.
- Counterfactual retraining.
- Federated unlearning.
- Large-scale hoặc multi-GPU execution.

## 6. Protocol đề xuất cho kết quả sơ bộ đầu tiên

```text
Model:          Qwen2.5-0.5B
Clients:        3
FL rounds:      1
LoRA rank:      4
LoRA targets:   q_proj, v_proj
Client data:    distinct synthetic textual facts
Prompts:        5–10 teacher-forced tracing prompts
Noise levels:   0, .01, .05, .1, .2, .5, 1.0
Seeds:          ít nhất 3
Metrics:        MAE, Spearman, Top-1, Top-k overlap, NDCG
```

Thứ tự thực hiện:

1. Khởi tạo base model và một LoRA adapter chung.
2. Train tuần tự ba client từ cùng base/adapter initialization.
3. Trích xuất true client updates và chạy FedAvg.
4. Tính oracle ProToken attribution bằng true updates.
5. Thêm controlled update noise và đo attribution degradation.
6. Chỉ khi bước 5 ổn định mới đưa FedAttr estimates vào ProToken.
7. Chỉ sau khi single-round giữ được ranking mới mở rộng sang multi-round.

## 7. Tiêu chí hoàn thành mốc tiếp theo

Mốc GPU đầu tiên hoàn thành khi repository tạo được một run có thể tái lập với:

- 3 trained LoRA client updates;
- 1 FedAvg global adapter;
- oracle score cho từng client/token/prompt;
- estimated score ở từng noise level;
- raw JSONL, config snapshot và seed;
- bảng/biểu đồ MAE, Spearman, Top-1 và NDCG;
- xác nhận tracer không dùng client full-model forward.

Kết quả này là cơ sở để quyết định tiếp tục FedAttr → ProToken, phát triển
noise-aware correction, hoặc chuyển trọng tâm sang attribution-preserving client
update estimation.
