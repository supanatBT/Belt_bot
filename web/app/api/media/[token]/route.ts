// GET /api/media/<token> — รูป drawing ที่บอทส่งกลับ (Space เสิร์ฟเฉพาะไฟล์ใต้ MEDIA_ROOTS)
import { SPACE_URL, notConfigured, spaceHeaders } from "@/lib/space";

export const runtime = "nodejs";

export async function GET(req: Request, { params }: { params: Promise<{ token: string }> }) {
  if (!SPACE_URL) return notConfigured();
  const { token } = await params;
  if (!/^[A-Za-z0-9_-]{8,64}$/.test(token)) return new Response("not found", { status: 404 });
  try {
    const res = await fetch(`${SPACE_URL}/media/${token}`, {
      headers: spaceHeaders(req),
      signal: AbortSignal.timeout(20_000),
    });
    if (!res.ok || !res.body) return new Response("not found", { status: 404 });
    const type = res.headers.get("content-type") || "";
    if (!type.startsWith("image/")) return new Response("not found", { status: 404 });
    return new Response(res.body, {
      headers: { "content-type": type, "cache-control": "public, max-age=86400" },
    });
  } catch {
    return new Response("unavailable", { status: 502 });
  }
}
