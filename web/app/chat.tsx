"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkBreaks from "remark-breaks";
import { prepareImage, ResizeError } from "@/lib/resize";

type Msg =
  | { id: string; role: "user"; text: string; image?: string }
  | { id: string; role: "bot"; text: string }
  | { id: string; role: "bot-image"; url: string; alt: string }
  | { id: string; role: "notice"; text: string; action?: "reset" };

const STORE_KEY = "beltbot.chat.v1";
const EXAMPLES = [
  "LF 820 K325 รับโหลดได้เท่าไหร่",
  "สายพานสำหรับไลน์เลาะกระดูกไก่ มีรุ่นไหนบ้าง",
  "MP80 C ทนอุณหภูมิได้เท่าไหร่",
];

const uid = () => Math.random().toString(36).slice(2, 10);

export default function Chat() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [sessionId, setSessionId] = useState("");
  const [text, setText] = useState("");
  const [image, setImage] = useState<{ file: File; url: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [preparing, setPreparing] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  // ── เก็บบทสนทนาไว้ในแท็บนี้ (รีเฟรชแล้วไม่หาย) — ไม่เก็บรูปของผู้ใช้ ──
  useEffect(() => {
    try {
      const saved = JSON.parse(sessionStorage.getItem(STORE_KEY) || "null");
      if (saved?.sessionId) setSessionId(saved.sessionId);
      if (Array.isArray(saved?.messages)) setMessages(saved.messages);
    } catch {}
  }, []);
  useEffect(() => {
    try {
      const keep = messages.map((m) => (m.role === "user" && m.image ? { ...m, image: undefined } : m));
      sessionStorage.setItem(STORE_KEY, JSON.stringify({ sessionId, messages: keep }));
    } catch {}
  }, [messages, sessionId]);
  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, busy]);

  const autoGrow = () => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 160) + "px";
  };

  const pickImage = async (f: File | undefined) => {
    if (!f) return;
    setPreparing(true);
    try {
      const small = await prepareImage(f);
      setImage((old) => {
        if (old) URL.revokeObjectURL(old.url);
        return { file: small, url: URL.createObjectURL(small) };
      });
    } catch (e) {
      const msg = e instanceof ResizeError ? e.message : "เปิดรูปนี้ไม่ได้ครับ";
      setMessages((m) => [...m, { id: uid(), role: "notice", text: msg }]);
    } finally {
      setPreparing(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  const send = useCallback(
    async (override?: string) => {
      const body = (override ?? text).trim();
      if (busy || preparing || (!body && !image)) return;
      const sent = image;
      setMessages((m) => [...m, { id: uid(), role: "user", text: body, image: sent?.url }]);
      setText("");
      setImage(null);
      requestAnimationFrame(autoGrow);
      setBusy(true);

      const form = new FormData();
      form.set("text", body);
      form.set("session_id", sessionId);
      if (sent) form.set("image", sent.file);

      try {
        const res = await fetch("/api/chat", { method: "POST", body: form });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          const notice: Msg = {
            id: uid(),
            role: "notice",
            text: data.message_th || "ส่งไม่สำเร็จครับ รบกวนลองใหม่อีกครั้ง",
            action: data.error === "session_full" ? "reset" : undefined,
          };
          setMessages((m) => [...m, notice]);
          if (data.error !== "session_full") {
            setText(body); // คืนข้อความให้ส่งใหม่ได้
            if (sent) setImage(sent);
          }
          return;
        }
        setSessionId(data.session_id);
        const incoming: Msg[] = (data.messages || []).map((m: any) =>
          m.type === "image"
            ? { id: uid(), role: "bot-image", url: m.url, alt: m.alt || "แบบสินค้า" }
            : { id: uid(), role: "bot", text: m.text },
        );
        setMessages((m) => [...m, ...incoming]);
      } catch {
        setMessages((m) => [
          ...m,
          { id: uid(), role: "notice", text: "เชื่อมต่อไม่ได้ครับ ตรวจอินเทอร์เน็ตแล้วลองใหม่" },
        ]);
        setText(body);
        if (sent) setImage(sent);
      } finally {
        setBusy(false);
        inputRef.current?.focus();
      }
    },
    [busy, preparing, text, image, sessionId],
  );

  const reset = async () => {
    const form = new FormData();
    form.set("session_id", sessionId);
    try {
      const res = await fetch("/api/session/reset", { method: "POST", body: form });
      const data = await res.json().catch(() => ({}));
      setSessionId(res.ok ? data.session_id : "");
    } catch {
      setSessionId("");
    }
    setMessages([]);
    setText("");
    setImage(null);
  };

  const empty = messages.length === 0;

  return (
    <div className="shell">
      <header className="top">
        <div className="brand">
          <span className="mark" aria-hidden>
            <i />
            <i />
            <i />
          </span>
          <div>
            <h1>ผู้ช่วยสินค้า</h1>
            <p>Movex โซ่ท็อปเชน · Modutech สายพานโมดูลาร์</p>
          </div>
        </div>
        {!empty && (
          <button className="ghost" onClick={reset} disabled={busy}>
            เริ่มแชตใหม่
          </button>
        )}
      </header>

      <main className="list" ref={listRef} aria-live="polite">
        {empty && (
          <div className="empty">
            <p>ถามสเปก รหัสรุ่น หรือแนบรูปโซ่ / สายพานให้ช่วยดูรุ่นได้เลยครับ</p>
            <div className="chips">
              {EXAMPLES.map((q) => (
                <button key={q} onClick={() => send(q)}>
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m) => {
          if (m.role === "user")
            return (
              <div key={m.id} className="row me">
                <div className="bubble user">
                  {m.image && <img src={m.image} alt="รูปที่แนบ" className="thumb" />}
                  {m.text && <div className="plain">{m.text}</div>}
                  {!m.text && !m.image && <div className="plain muted">(รูปที่แนบ)</div>}
                </div>
              </div>
            );
          if (m.role === "bot")
            return (
              <div key={m.id} className="row">
                <div className="bubble bot md">
                  <ReactMarkdown remarkPlugins={[remarkBreaks]}>{m.text}</ReactMarkdown>
                </div>
              </div>
            );
          if (m.role === "bot-image")
            return (
              <div key={m.id} className="row">
                <a className="bubble bot drawing" href={m.url} target="_blank" rel="noreferrer">
                  <img src={m.url} alt={m.alt} />
                </a>
              </div>
            );
          return (
            <div key={m.id} className="notice" role="status">
              <span>{m.text}</span>
              {m.action === "reset" && (
                <button className="link" onClick={reset}>
                  เริ่มแชตใหม่
                </button>
              )}
            </div>
          );
        })}
        {busy && (
          <div className="row">
            <div className="bubble bot typing" aria-label="กำลังตอบ">
              <span />
              <span />
              <span />
            </div>
          </div>
        )}
      </main>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          send();
        }}
      >
        {(image || preparing) && (
          <div className="attach">
            {preparing ? (
              <span className="muted">กำลังเตรียมรูป…</span>
            ) : (
              image && (
                <>
                  <img src={image.url} alt="รูปที่จะส่ง" />
                  <button type="button" className="link" onClick={() => setImage(null)}>
                    เอาออก
                  </button>
                </>
              )
            )}
          </div>
        )}
        <div className="bar">
          <button
            type="button"
            className="icon"
            onClick={() => fileRef.current?.click()}
            disabled={busy || preparing}
            aria-label="แนบรูป"
            title="แนบรูปสินค้า"
          >
            <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden>
              <path
                d="M4 7.5A2.5 2.5 0 0 1 6.5 5h1.8l1.2-1.6h5l1.2 1.6h1.8A2.5 2.5 0 0 1 20 7.5v9A2.5 2.5 0 0 1 17.5 19h-11A2.5 2.5 0 0 1 4 16.5z"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.6"
              />
              <circle cx="12" cy="12" r="3.4" fill="none" stroke="currentColor" strokeWidth="1.6" />
            </svg>
          </button>
          <input
            ref={fileRef}
            type="file"
            accept="image/jpeg,image/png,image/webp,image/heic,image/heif"
            hidden
            onChange={(e) => pickImage(e.target.files?.[0])}
          />
          <textarea
            ref={inputRef}
            rows={1}
            value={text}
            maxLength={1000}
            placeholder="พิมพ์คำถาม หรือแนบรูปสินค้า"
            onChange={(e) => {
              setText(e.target.value);
              autoGrow();
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                send();
              }
            }}
            onPaste={(e) => {
              const f = Array.from(e.clipboardData.files).find((x) => x.type.startsWith("image/"));
              if (f) {
                e.preventDefault();
                pickImage(f);
              }
            }}
          />
          <button type="submit" className="send" disabled={busy || preparing || (!text.trim() && !image)}>
            ส่ง
          </button>
        </div>
        <p className="fine">คำตอบมาจากแคตตาล็อก ราคา สต็อก และระยะเวลาส่งของ ฝ่ายขายเป็นผู้ยืนยัน</p>
      </form>
    </div>
  );
}
