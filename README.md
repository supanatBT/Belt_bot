# Belt Bot — ผู้ช่วยฝ่ายขายสายพาน Movex + Modutech

แชตบอตภาษาไทยตอบคำถามสเปกสินค้าสายพานลำเลียงสองแบรนด์

- **Movex** — โซ่/สายพานและสเตอร์ ค้นด้วยข้อความ และ**จำรุ่นจากรูปถ่าย**ได้ (ViT LoRA + SigLIP)
- **Modutech** — สายพานโมดูลาร์ 68 รุ่น (211 variant · 651 รายการการใช้งาน) ตอบจากแคตตาล็อกที่แปลงเป็น JSON

หลักการออกแบบสำคัญ: **เงื่อนไขความถูกต้องบังคับในโค้ด ไม่ฝากไว้ใน prompt**
ระบบตัดสินก่อนถึง LLM ว่าจะ *ตอบ / ถามกลับ / ปฏิเสธ / ส่งฝ่ายขาย* แล้วตัดข้อมูลที่ยังไม่ควรพูดออกจาก context
— ถ้าโมเดลไม่เห็นตัวเลข มันก็แต่งตัวเลขไม่ได้ Gemini มีหน้าที่แค่เรียบเรียงถ้อยคำจากข้อมูลที่ผ่านเกตแล้ว

> รายละเอียดการประกอบโฟลเดอร์ ประวัติปัญหาที่เคยเจอ และสิ่งที่ห้ามทำ อยู่ใน [`MANIFEST.md`](MANIFEST.md)

---

## สถาปัตยกรรม

```
เบราว์เซอร์
   │
   ▼
web/  (Next.js · Vercel)            เบราว์เซอร์ถือแค่ session_id
   │  /api/chat → แนบ PROXY_SECRET + IP จริงของผู้ใช้
   ▼
app.py  (FastAPI · HF Space · พอร์ต 7860)
   │  webapi/  rate limit · ตรวจรูป · เก็บ session ฝั่งเซิร์ฟเวอร์
   ▼
chatbot_v5.chat_interaction()
   │
   ├─ modutech_router     แยกแบรนด์ · จับรหัสสินค้าแบบ exact
   ├─ dialogue_state      ตัดสิน ตอบ / ถามกลับ / ปฏิเสธ / ส่งฝ่ายขาย  (template ตอบได้โดยไม่เรียก LLM)
   │
   ├─ Movex ─────► retriever_v4  (Qdrant + Jina v3 + ViT + SigLIP + knowledge_graph)
   │
   └─ Modutech ──► modutech_turn → modutech_gates (กรองข้อมูล) + modutech_kg (สายพาน↔สเตอร์)
                         │
                         ▼
                   Gemini (gemini-flash-latest) เรียบเรียงคำตอบ
   │
   ▼
pipeline_logger  บันทึกทุกขั้นลง logs/
```

---

## เริ่มใช้งาน

ต้องมี Python (มี `torch` · `transformers` · `peft` · `qdrant-client` · `gradio` · `fastapi` · `uvicorn` ·
`google-generativeai` · `networkx` · `python-dotenv` · `Pillow`) และ Node.js สำหรับหน้าเว็บ
เวอร์ชันที่เคยรันผ่านจดไว้ใน `env_lock.json`

**1. ตั้งค่า**

```
# .env  (รากโฟลเดอร์)
GEMINI_API_KEY=...
PROXY_SECRET=<สุ่มค่ายาว ๆ>

# web/.env.local
SPACE_URL=http://127.0.0.1:7860
PROXY_SECRET=<ค่าเดียวกับข้างบน>
```

**2. ตรวจก่อนรัน**

```powershell
python check_app.py        # ต้องขึ้น ✅ พร้อมรัน
python testsuite_120.py    # ชุดเกต ไม่ใช้เน็ต ไม่ใช้โควตา Gemini
```

**3. รัน** — ต้องสั่งจาก**รากโฟลเดอร์นี้** เพราะโค้ดอ้างพาธแบบ `./qdrant_db`

```powershell
.\run_app.ps1 -Python <path ของ python.exe>
```

หรือรันเองสองเทอร์มินัล

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 7860   # เทอร์มินัล 1 (โหลดโมเดลบน CPU ราว 1–2 นาที)
cd web ; npm install ; npm run dev                       # เทอร์มินัล 2 → http://localhost:3000
```

---

## หน้าที่ของแต่ละไฟล์

### แกนของบอต

| ไฟล์ | หน้าที่ |
|---|---|
| `chatbot_v5.py` | ตัวบอตหลัก โหลดโมเดลทั้งหมด (Jina v3, ViT LoRA, SigLIP) ต่อ Gemini และมี `chat_interaction()` ที่ทุกช่องทางเรียกใช้ รวมสองแบรนด์ไว้ในเทิร์นเดียว |
| `dialogue_state.py` | `DialogueManager` ตัดสินในโค้ดว่าเทิร์นนี้จะตอบ ถามกลับ ปฏิเสธนอกเรื่อง ปฏิเสธรูป หรือส่งฝ่ายขาย และจำบริบท (แบรนด์ รุ่น slot) ข้ามเทิร์น ไม่ import torch/qdrant จึงเทสต์ด้วย Python ล้วนได้ |
| `modutech_router.py` | แยกว่าคำถามเป็นของ Movex หรือ Modutech · แปลงรหัสที่พิมพ์มาหลายแบบให้เป็นรูปมาตรฐาน (`mp80-c` → `MP80C`) · resolve รหัสแบบ exact ก่อนค้นเวกเตอร์ · ค้นทั้งฐานสำหรับคำถามเชิงสำรวจ |
| `modutech_gates.py` | เกตกรองข้อมูล Modutech ก่อนส่งให้ LLM เรียงตามลำดับตายตัว: disqualifier → วัสดุ → ความกว้าง → โค้ง → ความแข็งแรง → แหล่งข้อมูล → อุณหภูมิ แล้วตัดเหลือเฉพาะฟิลด์ที่อนุญาต · มีคลาส `Catalogue` ที่โหลดและตรวจนับ JSON |
| `modutech_turn.py` | ประกอบ context + system prompt ของฝั่ง Modutech หนึ่งเทิร์น (ไม่เรียก LLM เอง) แยกไว้เพื่อเทสต์ได้ว่า "LLM เห็นอะไร" |
| `modutech_kg.py` | กราฟความสัมพันธ์ Modutech (สายพาน ↔ สเตอร์ · ซีรีส์ · วัสดุ) ตั้งใจ**ไม่เก็บตัวเลขสเปก** เพื่อไม่ให้ค่าที่เกตบล็อกรั่วออกทางกราฟ ตรวจลายนิ้วมือกับ JSON ทุกครั้งที่โหลด |
| `retriever_v4.py` | ตัวค้นฝั่ง Movex: รวมสัญญาณ ViT · SigLIP · Jina ด้วย confidence-weighted RRF กรองซีรีส์ใน Qdrant และขยายผลผ่านกราฟแบบ multi-hop |
| `knowledge_graph.py` | กราฟความเข้ากันได้ของ Movex (NetworkX เก็บใน SQLite `movex_kg.db`) |
| `pipeline_logger.py` | บันทึกทุกขั้นของการตอบลง `logs/` สามรูปแบบ: JSONL (ครบทุกฟิลด์) · CSV (เปิดใน Excel) · TXT (อ่านด้วยตา) |
| `sales_contact.py` | เบอร์โทรและอีเมลฝ่ายขาย แก้ที่ไฟล์นี้ที่เดียว ใช้ทั้งสองแบรนด์ |

### Web API (backend)

| ไฟล์ | หน้าที่ |
|---|---|
| `app.py` | FastAPI ครอบ `chat_interaction()` endpoint: `GET /health` · `POST /api/chat` · `POST /api/session/reset` · `GET /media/{token}` |
| `webapi/bot.py` | `BotAdapter` เรียกบอต · `SessionStore` เก็บประวัติแชตฝั่งเซิร์ฟเวอร์ต่อ session · `MediaRegistry` ออก token ให้รูปที่บอตส่งกลับ |
| `webapi/image_guard.py` | ตรวจรูปที่อัปโหลด: จำกัดขนาด · รับเฉพาะ JPEG/PNG/WEBP (ดูจากเนื้อไฟล์) · กัน decompression bomb · หมุนตาม EXIF · ย่อรูป |
| `webapi/ratelimit.py` | จำกัดจำนวนคำขอต่อ IP แบบ sliding window (ต่อนาที · ต่อชั่วโมง · รูปต่อชั่วโมง · รวมทั้งวัน) |
| `webapi/settings.py` | ค่าตั้งทั้งหมดอ่านจาก environment เช่น `PROXY_SECRET` · `RL_PER_MINUTE` · `MAX_CONCURRENT` · `SESSION_TTL_S` |

### หน้าเว็บ (`web/` — Next.js 16 + React 19)

| ไฟล์ | หน้าที่ |
|---|---|
| `app/chat.tsx` | หน้าจอแชต: พิมพ์ข้อความ แนบรูป แสดงคำตอบแบบ Markdown |
| `app/page.tsx` · `app/layout.tsx` · `app/globals.css` | หน้าหลัก โครงหน้า และสไตล์ |
| `app/api/chat/route.ts` | proxy ฝั่งเซิร์ฟเวอร์ไปยัง backend ตรวจขนาดข้อความ/รูปก่อนส่งต่อ |
| `app/api/session/reset/route.ts` | proxy สำหรับเริ่มบทสนทนาใหม่ |
| `app/api/media/[token]/route.ts` | proxy รูป drawing ที่บอตส่งกลับ |
| `lib/space.ts` | แนบ `PROXY_SECRET` · `SPACE_TOKEN` · IP จริงของผู้ใช้ — เบราว์เซอร์ไม่เห็นค่าเหล่านี้ |
| `lib/resize.ts` | ย่อรูปในเบราว์เซอร์ก่อนอัปโหลด |
| `next.config.ts` | ตั้ง security header |

### ข้อมูลและโมเดล

| ไฟล์/โฟลเดอร์ | หน้าที่ |
|---|---|
| `modutech_v30_1.json` | แคตตาล็อก Modutech ที่ใช้งานจริง (68 · 211 · 651) **ห้ามใช้ v30 หรือ v29** |
| `extracted_data_v3_machine_ready.json` | ข้อมูลสินค้า Movex |
| `qdrant_db/` | ฐานเวกเตอร์ Movex (collection `movex_text_chunks` และ `movex_images`) — ห้ามแก้ |
| `qdrant_modutech_db/` | ฐานเวกเตอร์ Modutech (collection `modutech_belt_chunks`, Jina 1024 มิติ) |
| `movex_kg.db` | กราฟ Movex (SQLite) |
| `modutech_kg.db` | กราฟ Modutech แบบไฟล์ ถ้าลายนิ้วมือไม่ตรงกับ JSON ระบบจะสร้างใหม่จาก JSON ให้เอง |
| `movex-lora-tuned-3/` | LoRA adapter ของ ViT สำหรับจำรุ่น Movex จากรูป (10 คลาส ใน `label_mapping.json`) |

### เครื่องมือประกอบและรัน

| ไฟล์ | หน้าที่ |
|---|---|
| `setup_app.ps1` | ประกอบโฟลเดอร์จากต้นทางที่รันได้ + zip ของเว็บ แล้วทับด้วย `latest/` (ดูคำเตือนด้านล่าง) และสุ่ม `PROXY_SECRET` ให้ |
| `check_app.py` | ตรวจ 6 หมวดก่อนรัน: python ถูกตัว · ไฟล์โค้ดครบ · ข้อมูล/โมเดลครบ · โค้ดเป็นรุ่นล่าสุด · เนื้อข้อมูลถูก · ค่าตั้งเว็บตรงกัน |
| `run_app.ps1` | ตรวจ → เปิด backend → รอพอร์ต 7860 พร้อม → เปิดหน้าเว็บ |
| `env_lock.json` | เวอร์ชันไลบรารีที่จดไว้ตอนรันผ่านครั้งแรก ใช้เทียบใน `check_app.py` |
| `patch_v30_1.py` | สคริปต์สร้าง `modutech_v30_1.json` จาก v30 (เติม notes 8 รุ่นจาก PDF · ตั้ง MD254 RR เกรด PPH เป็นค่ายังไม่ยืนยัน) |
| `MANIFEST.md` | คู่มือประกอบโฟลเดอร์ เหตุผลของผัง และรายการห้ามทำ |
| `latest/` | สำเนาไฟล์ที่แก้ 29–30 ก.ย. ไว้กันไฟล์รุ่นเก่าจาก zip ทับ |

### ชุดทดสอบ

| ไฟล์ | หน้าที่ |
|---|---|
| `testsuite_120.py` + `testsuite_120.csv` | ทดสอบชั้นที่ตัดสินด้วยโค้ด (router → dialogue → gate) 136 เคส ไม่ใช้ LLM ผลคงที่ ต้องได้ 136/136 |
| `testcases_modutech_136.csv` | 158 เทิร์นสำหรับรันกับบอตตัวเต็ม |
| `run_answer_eval.py` | harness ชั้น B — รันเคสผ่าน `chat_interaction()` + Gemini จริง แล้วตรวจถ้อยคำคำตอบ (คำที่ต้องมี/ห้ามมี · ค่า · หน่วย · ความยาว) |
| `testcases_graph.py` → `testcases_graph_relations.csv` | สร้างเคสทดสอบกราฟอัตโนมัติจากข้อมูลจริง ครอบทุก relation และตรวจว่าขนาดสเตอร์ไม่รั่ว |
| `test_modutech_kg.py` | ยืนยันว่ากราฟตรงกับ JSON ทุกเส้น ไม่มีตัวเลขสเปก และหยุดเมื่อข้อมูลขาด |
| `test_sprocket_flow.py` | ทดสอบเส้นทางคำถามสเตอร์ตั้งแต่ข้อความลูกค้าถึง context ที่ LLM เห็น |
| `test_increment_status.py` | ทดสอบตัวอ่านค่า "เพิ่มทีละ" จากหมายเหตุในแคตตาล็อก |
| `run3.json` · `graph_run1.json` · `graph_run2.json` · `graph_d_only.json` | ผลการรัน eval แต่ละรอบ (จำนวนผ่าน/ไม่ผ่านและรายละเอียดรายเทิร์น) |
| `logs/` | ตัวอย่าง log จริงจาก `pipeline_logger` |

---

## ⚠️ ข้อควรรู้ก่อนใช้ repo นี้

- **ไฟล์ที่รากใหม่กว่า `latest/`** — `chatbot_v5.py` · `modutech_gates.py` · `dialogue_state.py` ที่รากมีงานต่อกราฟ Modutech (30 ก.ย.)
  และการแก้ปัญหา `EC127SQZ24*PA` ที่ `latest/` ยังไม่มี ถ้ารัน `setup_app.ps1` ตอนนี้ ขั้น "ทับด้วย latest\\"
  จะย้อนงานเหล่านี้กลับ**โดยไม่มี error** ควรคัดลอกไฟล์ที่รากไปไว้ใน `latest/` ก่อน หรือข้ามขั้นนั้น
- **`.env.example` และ `web/.env.local.example` ยังไม่มีใน repo** ทั้งที่ `MANIFEST.md` และ `check_app.py` อ้างถึง ให้สร้างตามตัวอย่างในหัวข้อ "ตั้งค่า"
- **ห้าม commit `.env`** (มี `.gitignore` กันไว้แล้ว)
- ห้ามรัน `build_database.py` ในโฟลเดอร์นี้ — มันลบแล้วสร้าง `qdrant_db` ใหม่
- แก้ไฟล์ระหว่าง uvicorn รันอยู่ไม่มีผล ต้องรีสตาร์ต
- ยังไม่มี `requirements.txt` — สร้างด้วย `python -m pip freeze > requirements_lock.txt`
- โฟลเดอร์ `__pycache__/` ถูก commit ขึ้นมาด้วย ควรเพิ่มใน `.gitignore`
