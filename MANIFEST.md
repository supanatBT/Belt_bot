# โฟลเดอร์เดียวจบ — ในนั้นต้องมีอะไร และตรวจอะไรก่อนรัน

## ทำไมไฟล์บอตต้องอยู่ "แบน ๆ" ที่ราก

`chatbot_v5.py` อ้างพาธแบบ `./qdrant_db` · `./modutech_v30_1.json` · `./movex-lora-tuned-3`
พาธแบบนี้นับจาก **โฟลเดอร์ที่สั่งรัน** ไม่ใช่จากที่ไฟล์วางอยู่ ถ้าเอาไปซ้อนชั้นแล้วเปิดจากที่อื่น
จะขึ้น `FileNotFoundError` ทั้งที่ไฟล์อยู่ครบ — เสียเวลาไล่หาโดยไม่จำเป็น

มีแต่ `web/` ที่ยังต้องเป็นโฟลเดอร์ย่อย เพราะ Next.js ต้องมี `package.json` ของตัวเอง

---

## ผังที่ต้องได้

```
โฟลเดอร์นี้\
│
├─ โค้ด 10 ไฟล์ ─────────────── มาจาก 17_app_test ยกเว้น app.py ที่มาจาก zip
│   app.py                      FastAPI ที่หน้าเว็บเรียก          ← zip
│   chatbot_v5.py               ตัวบอต โหลดโมเดล ต่อ Gemini       ← latest\
│   retriever_v4.py             ค้นจาก Qdrant
│   knowledge_graph.py          กราฟความเข้ากันได้ของ Movex
│   pipeline_logger.py          บันทึกทุกขั้นของการตอบ
│   modutech_gates.py           เกตกรองคำตอบ Modutech             ← latest\
│   modutech_router.py          แยก Movex / Modutech
│   modutech_turn.py            ประกอบคำตอบหนึ่งเทิร์น
│   dialogue_state.py           จำบริบทข้ามเทิร์น                 ← latest\
│   sales_contact.py            เบอร์/อีเมลฝ่ายขายตัวจริง
│
├─ ข้อมูลและโมเดล 6 อย่าง ────── ยกมาจาก 17_app_test ทั้งหมด
│   qdrant_db\                  ฐานเวกเตอร์ Movex — ห้ามแก้
│   extracted_data_v3_machine_ready.json    ข้อมูลสินค้า Movex
│   movex_kg.db                  กราฟ Movex
│   movex-lora-tuned-3\          ViT LoRA ที่เทรนไว้ (ใช้ตอนส่งรูป)
│   qdrant_modutech_db\          ฐานเวกเตอร์ Modutech (collection แยก)
│   modutech_v30_1.json          68 · 211 · 651                   ← latest\
│
├─ ค่าตั้ง 2 ไฟล์ ──────────────
│   .env                        GEMINI_API_KEY + PROXY_SECRET
│   web\.env.local              SPACE_URL + PROXY_SECRET (ตรงกับข้างบน)
│
├─ web\ ──────────────────────── มาจาก zip ทั้งโฟลเดอร์
│   package.json · app\page.tsx · app\layout.tsx · app\globals.css
│
├─ เครื่องมือของโฟลเดอร์นี้ ─────
│   setup_app.ps1               ประกอบให้ครบในครั้งเดียว ลำดับถูกต้อง
│   check_app.py                ตรวจ 6 หมวดก่อนรัน
│   run_app.ps1                 เปิด backend รอจนพร้อม แล้วเปิดหน้าเว็บ
│   .env.example · web\.env.local.example
│
├─ latest\ ──────────────────── ของที่แก้ 29–30 ก.ย. ต้นฉบับอยู่ที่นี่
│   chatbot_v5.py · modutech_gates.py · dialogue_state.py · modutech_v30_1.json
│
└─ ชุดทดสอบ ──────────────────
    testsuite_120.py            136 เคส ชั้นเกต ไม่ใช้ LLM ไม่ใช้เน็ต
    testcases_modutech_136.csv  158 เทิร์น สำหรับรันกับบอตตัวเต็ม
    patch_v30_1.py              สคริปต์ที่สร้าง v30_1 จาก v30
    test_increment_status.py    เทสต์ย่อยของตัวอ่านค่าเพิ่มทีละ
```

`latest\` มีอยู่เพื่อกันปัญหาเดียว คือ **ลำดับการคัดลอก** zip ของเว็บ (27 ก.ย.) มี
`chatbot_v5.py` · `modutech_gates.py` · `dialogue_state.py` รุ่นก่อนการแก้อยู่ด้วย
ถ้าคัดลอกสลับลำดับ งานสองวันจะหาย **โดยไม่มี error** บอตยังรันได้ แค่กลับไปตอบผิดแบบเดิม
(เคยเกิดกับ `sales_contact.py` ที่เบอร์โทรหายเพราะ zip เก่าทับ) `setup_app.ps1` บังคับให้
`latest\` เป็นขั้นสุดท้ายเสมอ ทำมือแล้วลืมได้ สคริปต์ลืมไม่ได้

---

## ประกอบ

```powershell
cd <โฟลเดอร์นี้>
.\setup_app.ps1 -From "E:\Project2\modutech_artifacts\final\17_app_test" `
                -Zip  "<ที่เก็บ>\beltbot_web_deploy.zip"
```

ใช้ `17_app_test` เป็นต้นทางเพราะเป็นโฟลเดอร์ที่รันผ่านแล้ว ไฟล์ครบแน่กว่าไปหยิบทีละชื่อ
จากหลายโฟลเดอร์แล้วลืมอะไรไป สคริปต์จะสุ่ม `PROXY_SECRET` ให้ท้ายการทำงาน
เอาไปใส่ `.env` และ `web\.env.local` ให้เหมือนกัน แล้วใส่ `GEMINI_API_KEY` ใน `.env`

---

## ตรวจก่อนรัน

```powershell
$py = "E:\Project2\modutech_artifacts\final\8_buildchunks\.venv\Scripts\python.exe"
& $py check_app.py
```

ต้องขึ้น `✅ พร้อมรัน` ตัวนี้ตรวจหกหมวด

| หมวด | ตรวจอะไร | จับความผิดพลาดแบบไหน |
|---|---|---|
| 0 สภาพแวดล้อม | `sys.executable` · เวอร์ชัน transformers · torch · qdrant_client · peft · gradio เทียบกับ `env_lock.json` ที่จดไว้รอบแรก | ใช้ python ผิดตัว — 30 ก.ย. prompt ขึ้น `(.venv)` แต่เรียกตัวหลัก แล้วพังตั้งแต่ import |
| 1 โค้ด | ครบ 10 ไฟล์ที่ `chatbot_v5.py` import | ลืมคัดลอกไฟล์ไหนไป จะพังตอน import ไม่ใช่ตอนตอบ ดูเหมือนโค้ดเสีย |
| 2 ข้อมูลและโมเดล | หกอย่างมีจริงและมีขนาด · `qdrant_db` มี collection `movex_text_chunks` กับ `movex_images` | คัดลอก qdrant มาไม่ครบ — บอตขึ้นได้แต่ค้นไม่เจออะไรเลย ดูเหมือนโมเดลโง่ ไม่เหมือนไฟล์หาย |
| 3 เวอร์ชัน | หา 6 ข้อความเฉพาะในโค้ด เช่น `inch_basis` · `NO_INCREMENT_TH` · `FOLLOWUP_MAX_CHARS` · และ `MODUTECH_JSON` ต้องชี้ v30_1 | zip เก่าทับของใหม่ — ไม่ดูวันที่ไฟล์ เพราะวันที่เปลี่ยนตอนคัดลอกอยู่แล้ว |
| 4 เนื้อข้อมูล | นับ 68 · 211 · 651 · ทุกรุ่นมี `notes` · MD254 RR เกรด PPH เป็น `unconfirmed` · `sales_contact.py` ไม่ว่าง | เอา v30 หรือ v29 มาใช้ · เบอร์โทรหาย |
| 5–6 เว็บและค่าตั้ง | ไฟล์ Next.js สี่ตัว · มี node · `PROXY_SECRET` สองฝั่งตรงกัน · `SPACE_URL` ชี้ 127.0.0.1 · มี `GEMINI_API_KEY` | secret ไม่ตรง — หน้าเว็บขึ้นปกติแต่พิมพ์แล้ว error |

ที่เป็น `⚠` รันได้แต่ควรรู้ ที่เป็น `✗` ต้องแก้ก่อน

อยากตรวจลึกกว่านั้นให้รันชุดเกต ใช้เวลาไม่ถึงวินาที ไม่ใช้เน็ตและไม่ใช้โควตา Gemini

```powershell
& $py testsuite_120.py
```

ต้องได้ `รวม 136/136`

---

## รัน

```powershell
.\run_app.ps1 -Python $py
```

พิมพ์ `sys.executable` ให้ดูก่อน → ตรวจซ้ำ → เปิด backend → รอจนพอร์ต 7860 ตอบ
(โหลด Jina + ViT + SigLIP บน CPU ราว 1–2 นาที) → เปิดหน้าเว็บที่ `http://localhost:3000`

คุมเองก็ได้ สองเทอร์มินัล — **เทอร์มินัลแรกต้อง `cd` มาที่โฟลเดอร์นี้** ไม่ใช่เข้าไปในโฟลเดอร์ย่อย

```powershell
# เทอร์มินัล 1
& $py -m uvicorn app:app --host 127.0.0.1 --port 7860
# เทอร์มินัล 2
cd web ; npm run dev
```

---

## สี่คำถามที่ควรพิมพ์คุยก่อนถามนอกสคริปต์

ครอบทุกอย่างที่แก้ไป 29–30 ก.ย. ผ่านครบแปลว่าไฟล์ครบและถูกเวอร์ชันจริง ไม่ใช่แค่ผ่าน checker

1. `HC127 C มีความกว้างกี่นิ้ว` → 10–50 นิ้ว ครบ 21 ขนาด และ **ไม่ประกาศค่าเพิ่มทีละ**
2. `ขอเป็นหน่วยนิ้วได้ไหม` (ถามต่อทันที) → ต้องตอบได้ ไม่ปฏิเสธว่านอกเรื่อง
3. `หนาเท่าไหร่` (ถามต่อ) → 8 มม. (0.315 นิ้ว)
4. `MD254 RR ทนอุณหภูมิเท่าไหร่` → ตอบ `แห้ง` → 93 °C **พร้อมคำเตือนว่ายังไม่ยืนยัน**

---

## ห้ามทำในโฟลเดอร์นี้

- **ห้ามรัน `build_database.py`** ถ้ามันหลุดเข้ามา — มันลบและสร้าง collection ใหม่
  ชี้ไป `./qdrant_db` คือลบฐาน Movex ที่ใช้งานได้อยู่
- **ห้ามเอา `sales_contact.py` จาก zip มาทับ** — ของใน zip ว่าง ของจริงมีเบอร์กับอีเมล
- **ห้ามใช้ `modutech_v30.json` หรือ v29** — ค่าแรงดึง HC508 ผิด และ v30 ยังไม่มีคำเตือน MD254 RR
- **แก้ไฟล์ระหว่าง uvicorn รันอยู่ไม่มีผล** Python อ่านตอน import ครั้งเดียว ต้องรีสตาร์ต

## ยังไม่ได้ทำ

- **`bootstrap.py` ข้ามไป** มีไว้ดึงไฟล์ใหญ่จาก Hugging Face ตอน Space เริ่มทำงาน รันในเครื่องใช้ไฟล์ตรง ๆ
- **ImageRouter ยังไม่ต่อเข้า `chatbot_v5`** ส่งรูปยังเป็นของ Movex เหมือนเดิม
- **ยังไม่ตรึงเวอร์ชันไลบรารี** ควรทำก่อนส่งเล่ม ไม่งั้นกรรมการรันแล้วอาจเจอปัญหาเดียวกับ 30 ก.ย.

  ```powershell
  & $py -m pip freeze > requirements_lock.txt
  ```
