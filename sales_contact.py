"""
sales_contact.py — ช่องทางติดต่อฝ่ายขาย (แก้ที่ไฟล์นี้ที่เดียว ใช้ทั้ง Movex และ Modutech)

ใส่ค่าจริงในเครื่องหมายคำพูด ถ้าเว้นว่าง ("") บรรทัดนั้นจะไม่แสดงในคำตอบ
"""

SALES_PHONE = "086-3451403"    # เช่น "02-123-4567"
SALES_EMAIL = "Beltmactech@gmail.com"    # เช่น "sales@example.co.th"


def sales_contact_line_th() -> str:
    """บรรทัดช่องทางติดต่อ ต่อท้ายข้อความส่งฝ่ายขาย — คืน "" ถ้ายังไม่ได้ใส่ค่า"""
    parts = []
    if SALES_PHONE.strip():
        parts.append(f"โทร {SALES_PHONE.strip()}")
    if SALES_EMAIL.strip():
        parts.append(f"อีเมล {SALES_EMAIL.strip()}")
    return ("ติดต่อฝ่ายขาย: " + " · ".join(parts)) if parts else ""
