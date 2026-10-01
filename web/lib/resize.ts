// lib/resize.ts — ย่อรูปในเบราว์เซอร์ก่อนส่ง: ด้านยาวสุด 1600 px, JPEG 0.85
// ประหยัดเน็ตผู้ใช้ + ไม่ชนเพดาน body ของ Vercel · หมุนตาม EXIF และตัด EXIF (พิกัด GPS) ทิ้งไปในตัว

export const MAX_SIDE = 1600;
export const MAX_INPUT_BYTES = 20 * 1024 * 1024;
const ACCEPT = ["image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"];

export class ResizeError extends Error {}

export async function prepareImage(file: File): Promise<File> {
  if (!ACCEPT.includes(file.type) && !/\.(jpe?g|png|webp|heic|heif)$/i.test(file.name)) {
    throw new ResizeError("รองรับเฉพาะรูป JPG, PNG, WEBP ครับ");
  }
  if (file.size > MAX_INPUT_BYTES) throw new ResizeError("รูปใหญ่เกิน 20 MB ครับ");

  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  } catch {
    throw new ResizeError("เปิดรูปนี้ไม่ได้ครับ ลองถ่ายใหม่หรือบันทึกเป็น JPG");
  }
  const scale = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
  const w = Math.round(bitmap.width * scale);
  const h = Math.round(bitmap.height * scale);
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new ResizeError("เบราว์เซอร์นี้ย่อรูปไม่ได้ครับ");
  ctx.fillStyle = "#fff"; // PNG โปร่งใส → พื้นขาว (เหมือน convert("RGB") ฝั่งบอท)
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(bitmap, 0, 0, w, h);
  bitmap.close();
  const blob: Blob | null = await new Promise((r) => canvas.toBlob(r, "image/jpeg", 0.85));
  if (!blob) throw new ResizeError("ย่อรูปไม่สำเร็จครับ");
  return new File([blob], "upload.jpg", { type: "image/jpeg" });
}
