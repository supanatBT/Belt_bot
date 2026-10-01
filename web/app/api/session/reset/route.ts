import { SPACE_URL, notConfigured, relay, spaceHeaders, upstreamDown } from "@/lib/space";

export const runtime = "nodejs";

export async function POST(req: Request) {
  if (!SPACE_URL) return notConfigured();
  let sessionId = "";
  try {
    sessionId = String((await req.formData()).get("session_id") ?? "").slice(0, 64);
  } catch {}
  const body = new FormData();
  body.set("session_id", sessionId);
  try {
    const res = await fetch(`${SPACE_URL}/api/session/reset`, {
      method: "POST",
      headers: spaceHeaders(req),
      body,
      signal: AbortSignal.timeout(15_000),
    });
    return relay(res);
  } catch {
    return upstreamDown();
  }
}
