# run_app.ps1 — เปิด backend และหน้าเว็บ จากโฟลเดอร์เดียว
#
# ต้องสั่ง uvicorn จากโฟลเดอร์นี้ เพราะพาธในโค้ดเป็นแบบ ./qdrant_db, ./modutech_v30_1.json
# ซึ่งนับจากโฟลเดอร์ที่สั่งรัน ไม่ใช่จากที่ไฟล์วางอยู่ สคริปต์นี้ cd ให้เองแล้ว
#
# ระบุ interpreter ให้ชัดดีกว่าพึ่ง activate
#   30 ก.ย. prompt ขึ้น (.venv) แต่ python ไปเรียกตัวหลักที่มี transformers คนละเวอร์ชัน
#   แล้วพังตั้งแต่ตอน import — เสียเวลาไล่หาเพราะไม่มีใครดู sys.executable
#
# ตัวอย่าง
#   .\run_app.ps1 -Python "E:\Project2\modutech_artifacts\final\8_buildchunks\.venv\Scripts\python.exe"

param([string]$Python = "python")

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

Write-Host "== interpreter ที่จะใช้ ==" -ForegroundColor Cyan
& $Python -c "import sys; print(sys.executable)"
if ($LASTEXITCODE -ne 0) { Write-Host "เรียก python ไม่ได้: $Python" -ForegroundColor Red; exit 1 }

Write-Host "`n== ตรวจก่อนรัน ==" -ForegroundColor Cyan
Push-Location $root
& $Python "$root\check_app.py"
$rc = $LASTEXITCODE
Pop-Location
if ($rc -ne 0) { exit 1 }

# ── backend ──────────────────────────────────────────────────
Write-Host "`n== เปิด backend (พอร์ต 7860) ==" -ForegroundColor Cyan
$cmd = "cd '$root'; " +
       "Get-Content .env | Where-Object { `$_ -and `$_ -notmatch '^#' } | ForEach-Object { " +
       "`$k,`$v = `$_ -split '=',2; [Environment]::SetEnvironmentVariable(`$k.Trim(), `$v.Trim()) }; " +
       "& '$Python' -m uvicorn app:app --host 127.0.0.1 --port 7860"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $cmd

Write-Host "รอ backend พร้อม (โหลด Jina + ViT + SigLIP บน CPU ราว 1-2 นาที)..." -NoNewline
$ready = $false
foreach ($i in 1..180) {
    Start-Sleep -Seconds 1
    try {
        $c = New-Object Net.Sockets.TcpClient
        $c.Connect("127.0.0.1", 7860)
        $c.Close(); $ready = $true; break
    } catch { Write-Host "." -NoNewline }
}
if (-not $ready) {
    Write-Host "`n❌ backend ไม่ขึ้นภายใน 3 นาที — ดู error ในหน้าต่างที่เปิดไว้" -ForegroundColor Red
    exit 1
}
Write-Host " พร้อมแล้ว" -ForegroundColor Green

# ── หน้าเว็บ ─────────────────────────────────────────────────
Write-Host "`n== เปิดหน้าเว็บ (พอร์ต 3000) ==" -ForegroundColor Cyan
if (-not (Test-Path "$root\web\node_modules")) {
    Write-Host "ยังไม่ได้ npm install — กำลังติดตั้ง (ครั้งแรกใช้เวลาสักครู่)"
    Push-Location "$root\web"; npm install; Pop-Location
}
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\web'; npm run dev"

Start-Sleep -Seconds 4
Write-Host "`nเปิดเบราว์เซอร์ที่ http://localhost:3000" -ForegroundColor Green
Start-Process "http://localhost:3000"
