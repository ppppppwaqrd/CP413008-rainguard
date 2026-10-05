# RainGuard

ระบบพยากรณ์ว่าวันพรุ่งนี้ฝนจะตกหรือไม่ เพื่อให้เกษตรกรตัดสินใจพ่นสาร ให้น้ำ หรือเลื่อนเก็บเกี่ยว

รายวิชา CP413008 Machine Learning Engineering for Production ภาคเรียนที่ 1/2569

กลุ่ม 2 ชื่อกลุ่ม **หลอ**

- จักรภัทร เวียงสิมมา 673380308-6 หมู่ 1
- สรวิศ สำราญบึงแก 673380348-4 หมู่ 1
- พรหมพัฒน ศิริภัคกุลวัฒน์ 673380328-0 หมู่ 2

สาขาที่ส่งคือ `main` รายงานฉบับเต็มอยู่ที่ [`report/REPORT.md`](report/REPORT.md)

## ผลที่ใช้ส่ง

โมเดลที่ให้บริการคือ hist_gb เวอร์ชัน 2

- ROC-AUC 0.8877 และ recall ของคลาสฝน 0.8246 บนชุดทดสอบถึงวันที่ 30 มกราคม 2026
- ต้นทุนพลาดฝน 5 หน่วยต่อเตือนผิด 1 หน่วย อยู่ที่ 15,004 ต่ำกว่าการไม่เตือนเลย (43,220) และการเตือนทุกวัน (33,375)
- API ใน Docker: p50 41.2 ms, p95 68.1 ms, throughput 20.30 คำขอต่อวินาที ผ่าน SLO ที่ประกาศไว้
- ปริมาณฝนติดลบถูกปฏิเสธก่อนถึงโมเดล API ตอบ 422

ตัวเลขอยู่ใน `reports/experiments.json`, `reports/slo.json`, `reports/rollback.json`

## วิธีตรวจงานนี้บน GitHub

งานบน `main` ผ่าน GitHub Actions แล้ว ทั้งคุณภาพโค้ด สัญญาข้อมูล และคุณภาพโมเดล

Pull Request ที่ยังเปิดอยู่ชื่อ **Show CI rejecting a loosened rainfall contract** เป็นหลักฐานรอบที่ข้อมูลเสียแล้ว CI ไม่ผ่าน อย่า merge อันนั้น ถ้า merge ขอบล่างของปริมาณฝนจะหลุด และ `main` จะยอมรับค่าติดลบ

## รันจากเครื่องเปล่า

ต้องมี Python 3.11 ขึ้นไป ถ้าจะเปิด API ทั้งสแตกต้องมี Docker

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python scripts/download_data.py
python -m rainguard.cli run
```

คำสั่ง `python -m rainguard.cli run` เดินจากไฟล์ดิบถึงโมเดลที่พร้อมเสิร์ฟ: ตรวจสัญญาข้อมูล แบ่งตามเวลา เทรน 3 โมเดล บันทึก MLflow ผ่านด่าน blessing แล้วคัดลอกโมเดลขึ้น `artifacts/serving`

ไฟล์ข้อมูลดิบประมาณ 30 MB, `mlruns` และ `artifacts` ไม่ได้อยู่ใน git ผู้รับต้องดาวน์โหลดข้อมูลแล้วรันคำสั่งด้านบน

ถ้าส่งไฟล์เสีย ระบบหยุดก่อนเทรนและเขียน `alerts/validation_error.txt`

```powershell
python scripts/make_bad_data.py
python -m rainguard.cli validate --data data/bad_weather.csv
```

## ให้บริการ

```powershell
docker compose up --build
```

- API: http://127.0.0.1:8000/health
- MLflow: http://127.0.0.1:5000
- Prometheus: http://127.0.0.1:9090
- Grafana: http://127.0.0.1:3000 ชื่อผู้ใช้ `admin` รหัส `admin`

วัด latency ของคอนเทนเนอร์ที่เปิดอยู่แล้ว:

```powershell
python scripts/loadtest.py --base http://127.0.0.1:8000
```

Airflow ใช้โปรไฟล์แยก DAG `rainguard_train` เรียกฟังก์ชันชุดเดียวกับคำสั่ง `run`

```powershell
docker compose --profile airflow up --build airflow
```

## แผนที่โฟลเดอร์

- `report/REPORT.md` รายงาน
- `reports/` ตัวเลขและหลักฐาน CI
- `examples/predict.json` คำขอปกติ `examples/bad_predict.json` คำขอที่ต้องได้ 422
- `src/rainguard/` สัญญาข้อมูล ตัวแปลงร่วม และโค้ดเทรน
- `serving/app.py` API
- `dags/rain_train.py` DAG
- `monitoring/` Prometheus กับ Grafana
- `scripts/` ดาวน์โหลดข้อมูล วัด latency เทรนใหม่ และย้อนกลับโมเดล
- `tests/` ชุดที่ GitHub Actions รัน
- `.github/workflows/ci.yml` ตรวจสามด้าน: ruff, สัญญาข้อมูล, คุณภาพโมเดล

ใช้ Cursor ช่วยเขียนโค้ดและร่างรายงาน รายละเอียดอยู่ในท้าย `report/REPORT.md`
