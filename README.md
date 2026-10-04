# UA-DETRAC NMS-free training project

Đây là project độc lập để mang sang máy khác/Kaggle train model nền 4 lớp trên
UA-DETRAC. Cấu trúc và code train giữ theo pipeline hiện tại; không cần backend,
database hay venv của repo gốc. Dataset được mount thành folder riêng trong
Kaggle và không được copy vào project này.

## Cấu trúc

```text
model_developement/object_detection/   model, loss, dataset loader
demo-od/core/                          train pipeline, validation và metrics
demo-od/runners/run_train.sh           runner chính
demo-od/run_train.sh                   wrapper tương thích
```

UA-DETRAC cần có bốn class trong `metadata.json`: `0: car`, `1: bus`, `2: van`,
`3: others`; thư mục dataset cần có `images/`, `labels/`, `metadata.json`.

## Chạy trên Kaggle

1. Tạo Notebook với accelerator GPU và attach dataset UA-DETRAC.
2. Đưa thư mục project này vào working directory của Notebook (hoặc clone/copy
   project vào `/kaggle/working`). Kaggle input chỉ đọc được; output phải ghi vào
   `/kaggle/working`.
3. Đặt `OD_DATASET_DIR` thành thư mục dataset đã attach. Pipeline tự tìm ba thành
   phần cần thiết ở thư mục đó hoặc tối đa hai cấp thư mục con.

```bash
cd /kaggle/working/kaggle_ua_detrac_nmsfree
set -o pipefail
OD_DATASET_DIR="/kaggle/input/ua-detrac" \
OD_OUTPUT_DIR=/kaggle/working/od_output \
OD_EPOCHS=60 OD_BATCH_SIZE=8 OD_WORKERS=2 \
bash demo-od/run_train.sh 2>&1 | tee /kaggle/working/train.log
```

Model PyTorch/CUDA được Kaggle cung cấp sẵn. Nếu thiếu thư viện phụ trợ, cài:

```bash
pip install -r requirements-kaggle.txt
```

Không cài lại `torch`/`torchvision` một cách mù quáng trong Kaggle vì cần khớp
bản CUDA của runtime. Nếu gặp OOM, giảm `OD_BATCH_SIZE` (ví dụ 4 hoặc 2), rồi
giảm `OD_WORKERS` về 0 hoặc 1.

## Theo dõi và kết quả

- `train.log`: loss train, validation mAP50/mAP50-95, precision và recall.
- `od_output/<job-id>_preview/`: ảnh preview của các ảnh validation cố định; cứ
  mỗi 2.000 ảnh train sẽ tạo snapshot mới và chỉ giữ lại 3 snapshot gần nhất.
- `od_output/<job-id>_checkpoint/`: checkpoint trong lúc train; model cuối cùng
  được chọn theo validation mAP50-95 tốt nhất.
- Khi pipeline hoàn thành: `od_output/<job-id>/` và `od_output/<job-id>.zip`.
- Có thể đổi tần suất preview: `OD_PREVIEW_EVERY_IMAGES=5000`; đặt `0` để tắt.
- Có thể đổi số epoch, batch, worker, device qua `OD_EPOCHS`, `OD_BATCH_SIZE`,
  `OD_WORKERS`, `OD_DEVICE`.

Split validation được nhóm theo sequence trong tên ảnh UA-DETRAC để các frame
liền nhau không bị chia sang cả train và validation. Dataset Kaggle chỉ được đọc,
không bị sửa hoặc sao chép vào bộ nhớ làm việc.
