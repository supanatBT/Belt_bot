# setup_app.ps1 — ประกอบโฟลเดอร์นี้ให้ครบในครั้งเดียว
#
# ทำไมต้องมีสคริปต์ ไม่ทำมือ
#   ลำดับการคัดลอกสำคัญกว่าที่คิด zip ของเว็บ (27 ก.ย.) มี chatbot_v5.py ·
#   modutech_gates.py · dialogue_state.py รุ่นก่อนการแก้วันที่ 29–30 ก.ย. อยู่ด้วย
#   ถ้าคัดลอกสลับลำดับ งานที่แก้ไปจะหายเงียบ ๆ ไม่มี error บอตยังรันได้ แค่ตอบผิดแบบเดิม
#   (เคยเกิดกับ sales_contact.py ที่เบอร์โทรหายไปเพราะ zip เก่าทับ)
#   สคริปต์นี้บังคับลำดับ zip → โฟลเดอร์ที่รันได้ → latest\ เป็นขั้นสุดท้ายเสมอ
#
# ตัวอย่าง
#   .\setup_app.ps1 -From "E:\Project2\modutech_artifacts\final\17_app_test" `
#                   -Zip  "E:\Downloads\beltbot_web_deploy.zip"

param(
    # โฟลเดอร์ที่บอตรันได้อยู่จริง ใช้เป็นต้นทางของโค้ด ฐานข้อมูล และโมเดล
    [Parameter(Mandatory = $true)][string]$From,
    # beltbot_web_deploy.zip — ที่มา app.py และหน้าเว็บ Next.js
    [Parameter(Mandatory = $true)][string]$Zip
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

foreach ($p in @($From, $Zip)) {
    if (-not (Test-Path $p)) { Write-Host "ไม่พบ $p" -ForegroundColor Red; exit 1 }
}
if ((Resolve-Path $From).Path -eq (Resolve-Path $root).Path) {
    Write-Host "-From ชี้มาที่โฟลเดอร์นี้เอง" -ForegroundColor Red; exit 1
}

# ── 1. หน้าเว็บกับ app.py จาก zip ─────────────────────────────
Write-Host "1/4 แตก zip ของเว็บ" -ForegroundColor Cyan
$tmp = Join-Path $env:TEMP ("beltbot_" + [guid]::NewGuid().ToString("N").Substring(0, 8))
Expand-Archive -Path $Zip -DestinationPath $tmp -Force

# zip แยกเป็น space\ (backend) กับ web\ (Next.js) — ผังนี้เอา backend มาไว้ที่ราก
$spaceDir = Get-ChildItem $tmp -Recurse -Directory -Filter space | Select-Object -First 1
$webDir = Get-ChildItem $tmp -Recurse -Directory -Filter web | Select-Object -First 1
if ($spaceDir) { Copy-Item "$($spaceDir.FullName)\*" $root -Recurse -Force }
else { Write-Host "   ⚠ ไม่พบ space\ ใน zip — คัดลอกทุกอย่างมาที่รากแทน" -ForegroundColor Yellow
       Get-ChildItem $tmp -File | Copy-Item -Destination $root -Force }
if ($webDir) {
    New-Item -ItemType Directory -Force -Path "$root\web" | Out-Null
    Copy-Item "$($webDir.FullName)\*" "$root\web" -Recurse -Force
} else { Write-Host "   ⚠ ไม่พบ web\ ใน zip — หน้าเว็บจะไม่ครบ" -ForegroundColor Yellow }
Remove-Item $tmp -Recurse -Force

# ── 2. โค้ด ฐานข้อมูล โมเดล จากโฟลเดอร์ที่รันได้อยู่ ────────────
# -Exclude app.py กันไม่ให้ทับ backend ของเว็บที่เพิ่งได้จาก zip
# ขั้นนี้กินเวลาสุด เพราะมี qdrant สองตัวและโฟลเดอร์ ViT LoRA
Write-Host "2/4 คัดลอกจาก $From (ใหญ่ ใช้เวลาสักครู่)" -ForegroundColor Cyan
Copy-Item "$From\*" $root -Recurse -Force -Exclude app.py

# ── 3. ของที่แก้ 29–30 ก.ย. — ขั้นสุดท้ายเสมอ ──────────────────
Write-Host "3/4 ทับด้วยไฟล์ใน latest\" -ForegroundColor Cyan
if (-not (Test-Path "$root\latest")) { Write-Host "   ไม่พบ latest\" -ForegroundColor Red; exit 1 }
Get-ChildItem "$root\latest" -File | ForEach-Object {
    Copy-Item $_.FullName $root -Force
    Write-Host "   ทับ $($_.Name)"
}

# ── 4. ค่าตั้ง ────────────────────────────────────────────────
Write-Host "4/4 ค่าตั้ง" -ForegroundColor Cyan
if (-not (Test-Path "$root\.env")) {
    Copy-Item "$root\.env.example" "$root\.env"
    Write-Host "   สร้าง .env แล้ว — ยังต้องใส่ GEMINI_API_KEY และ PROXY_SECRET"
} else { Write-Host "   .env มีอยู่แล้ว ไม่แตะ" }
if (-not (Test-Path "$root\web\.env.local")) {
    Copy-Item "$root\web\.env.local.example" "$root\web\.env.local"
    Write-Host "   สร้าง web\.env.local แล้ว — ยังต้องใส่ PROXY_SECRET ให้ตรงกับ .env"
} else { Write-Host "   web\.env.local มีอยู่แล้ว ไม่แตะ" }

$secret = -join ((48..57) + (65..90) + (97..122) | Get-Random -Count 43 | ForEach-Object { [char]$_ })
Write-Host "`nค่า PROXY_SECRET ที่สุ่มให้ (ใส่ทั้งสองไฟล์ให้เหมือนกัน):" -ForegroundColor Green
Write-Host "   $secret"
Write-Host "`nเสร็จแล้วรัน:  python check_app.py" -ForegroundColor Green
