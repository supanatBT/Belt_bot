// lib/space.ts — เรียก backend บน Hugging Face Space จากฝั่งเซิร์ฟเวอร์ของ Vercel เท่านั้น
// เบราว์เซอร์ไม่เห็น SPACE_URL / PROXY_SECRET / SPACE_TOKEN
import "server-only";

export const SPACE_URL = (process.env.SPACE_URL || "").replace(/\/+$/, "");

export function spaceHeaders(req: Request): Record<string, string> {
  const h: Record<string, string> = {};
  if (process.env.PROXY_SECRET) h["x-proxy-secret"] = process.env.PROXY_SECRET;
  // Space แบบ private ต้องใช้ token ของ HF
  if (process.env.SPACE_TOKEN) h["authorization"] = `Bearer ${process.env.SPACE_TOKEN}`;
  // Vercel ใส่ IP จริงของผู้ใช้เป็นตัวแรกใน x-forwarded-for (ผู้ใช้ปลอมค่านี้ผ่าน Vercel ไม่ได้)
  const ip =
    req.headers.get("x-real-ip") ||
    (req.headers.get("x-forwarded-for") || "").split(",")[0].trim() ||
    "unknown";
  h["x-client-ip"] = ip;
  return h;
}

export function notConfigured() {
  return Response.json(
    { error: "not_configured", message_th: "ยังไม่ได้ตั้งค่าเซิร์ฟเวอร์ (SPACE_URL)" },
    { status: 500 },
  );
}

export function upstreamDown() {
  return Response.json(
    {
      error: "upstream",
      message_th: "ติดต่อระบบตอบคำถามไม่ได้ชั่วคราวครับ อาจกำลังเริ่มต้นใหม่ รบกวนลองอีกครั้งในอีกสักครู่",
    },
    { status: 502 },
  );
}

/** ส่งต่อคำตอบจาก Space (สถานะ + Retry-After) */
export async function relay(res: Response, rewrite?: (body: any) => any) {
  let body: any;
  try {
    body = await res.json();
  } catch {
    // Space ที่กำลัง build / หลับ มักตอบเป็น HTML
    return upstreamDown();
  }
  const headers: Record<string, string> = {};
  const ra = res.headers.get("retry-after");
  if (ra) headers["retry-after"] = ra;
  return Response.json(rewrite && res.ok ? rewrite(body) : body, { status: res.status, headers });
}
