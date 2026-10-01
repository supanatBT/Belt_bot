// POST /api/chat — proxy ไปที่ Space: แนบ PROXY_SECRET + IP จริง · ตรวจขนาดก่อนส่งต่อ
import { SPACE_URL, notConfigured, relay, spaceHeaders, upstreamDown } from "@/lib/space";

export const runtime = "nodejs";
export const maxDuration = 60;

const MAX_IMAGE_BYTES = 4 * 1024 * 1024; // ฟังก์ชัน Vercel รับ body ได้ราว 4.5 MB
const MAX_TEXT = 1000;

export async function POST(req: Request) {
  if (!SPACE_URL) return notConfigured();

  let form: FormData;
  try {
    form = await req.formData();
  } catch {
    return Response.json({ error: "bad_request", message_th: "ส่งข้อมูลไม่ถูกต้อง" }, { status: 400 });
  }
  const text = String(form.get("text") ?? "").slice(0, MAX_TEXT + 1);
  const sessionId = String(form.get("session_id") ?? "").slice(0, 64);
  const image = form.get("image");

  if (text.length > MAX_TEXT) {
    return Response.json(
      { error: "text_too_long", message_th: `ข้อความยาวเกิน ${MAX_TEXT} ตัวอักษรครับ` },
      { status: 400 },
    );
  }
  const out = new FormData();
  out.set("text", text);
  out.set("session_id", sessionId);
  if (image instanceof File && image.size > 0) {
    if (image.size > MAX_IMAGE_BYTES) {
      return Response.json(
        { error: "image_too_large", message_th: "รูปใหญ่เกิน 4 MB ครับ" },
        { status: 400 },
      );
    }
    out.set("image", image, image.name || "upload.jpg");
  }

  let res: Response;
  try {
    res = await fetch(`${SPACE_URL}/api/chat`, {
      method: "POST",
      headers: spaceHeaders(req),
      body: out,
      signal: AbortSignal.timeout(55_000),
    });
  } catch {
    return upstreamDown();
  }
  // รูปที่บอทส่งกลับ: /media/<token> → /api/media/<token> (ผ่าน proxy นี้)
  return relay(res, (body) => ({
    ...body,
    messages: (body.messages || []).map((m: any) =>
      m.type === "image" && typeof m.url === "string" && m.url.startsWith("/media/")
        ? { ...m, url: `/api${m.url}` }
        : m,
    ),
  }));
}
