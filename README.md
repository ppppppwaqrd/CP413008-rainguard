# RainGuard

ระบบพยากรณ์ความเสี่ยงฝนวันพรุ่งนี้ สำหรับตัดสินใจพ่นสาร ให้น้ำ หรือเลื่อนเก็บเกี่ยว

กลุ่มหลอ รายวิชา CP413008 — จักรภัทร เวียงสิมมา, สรวิศ สำราญบึงแก, พรหมพัฒน ศิริภัคกุลวัฒน์

## รันจากเครื่องเปล่า

ต้องมี Python 3.11 ขึ้นไป และ Docker ถ้าจะเปิด API กับ Grafana

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python scripts/download_data.py
python -m rainguard.cli run
```

คำสั่ง `python -m rainguard.cli run` คือเส้นทางเดียวจากไฟล์ดิบถึงโมเดลที่พร้อมเสิร์ฟ: ตรวจสัญญาข้อมูล แบ่งตามเวลา เทรน 3 โมเดล บันทึก MLflow ผ่านด่าน blessing แล้วคัดลอกโมเดลขึ้น `artifacts/serving`

ถ้าส่งไฟล์เสีย ระบบหยุดก่อนเทรนและเขียน `alerts/validation_error.txt`

```powershell
python scripts/make_bad_data.py
python -m rainguard.cli validate --data data/bad_weather.csv
```

## ให้บริการและวัด SLO

```powershell
python scripts/loadtest.py
python scripts/retrain_cycle.py
python scripts/rollback.py
```

`loadtest.py` เปิด API ที่พอร์ต 8000 วัด latency แล้วปิดโปรเซสเอง

เปิดทั้งสแตก (API, MLflow, Prometheus, Grafana):

```powershell
docker compose up --build
```

- API: http://127.0.0.1:8000/health
- MLflow: http://127.0.0.1:5000
- Prometheus: http://127.0.0.1:9090
- Grafana: http://127.0.0.1:3000 (admin / admin)

Airflow ใช้โปรไฟล์แยก เพราะอิมเมจใหญ่กว่า:

```powershell
docker compose --profile airflow up --build airflow
```

DAG `rainguard_train` เรียกฟังก์ชันชุดเดียวกับคำสั่ง `run` และตั้งเวลาทุกวันจันทร์ 06:00

## ทดสอบ

```powershell
ruff check src tests serving scripts dags
pytest
python scripts/ci_gate.py
```

หลักฐานที่ข้อมูลเสียแล้วไม่ผ่านอยู่ที่ `reports/ci_fail.txt` หลักฐานรอบที่ผ่านอยู่ที่ `reports/ci_pass.txt`

รายงานฉบับเต็มอยู่ที่ `report/REPORT.md`

## ผลที่วัดได้บนเครื่องนี้

- hist_gb ขึ้น Production เวอร์ชัน 2, ROC-AUC 0.8877, recall 0.8246 บนชุดทดสอบถึงวันที่ 2026-01-30
- API ในคอนเทนเนอร์: p50 41.2 ms, p95 68.1 ms, throughput 20.30 คำขอต่อวินาที (`reports/slo.json`)
- รอบ uvicorn บนโฮสต์: p50 62.2 ms, p95 84.2 ms, throughput 15.36 คำขอต่อวินาที (`reports/slo_host.json`)
- ย้อนกลับทะเบียนจากเวอร์ชัน 4 ไปเวอร์ชัน 2 แล้ว `/health` ตอบเวอร์ชัน 2

วัด SLO รอบคอนเทนเนอร์ด้วย `python scripts/loadtest.py --base http://127.0.0.1:8000` ขณะที่ `docker compose up --build` เปิดบริการอยู่

