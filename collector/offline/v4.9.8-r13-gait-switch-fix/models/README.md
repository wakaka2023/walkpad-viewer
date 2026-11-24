# YOLO 姿态权重

`yolo26s-pose.pt`（24,151,790 字节）不随 Git 提交。可以从原交付 ZIP 提取，也可以下载 [Ultralytics v8.4.0 Release 附件](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-pose.pt)；该附件公布的 SHA-256 与原包一致。目标路径是离线包版本目录下的 `models/yolo26s-pose.pt`。

**从版本目录运行（PowerShell）：**

```powershell
curl.exe --fail --location --output models/yolo26s-pose.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-pose.pt
if ($LASTEXITCODE -ne 0) { throw 'Weight download failed' }
if ((Get-FileHash -Algorithm SHA256 models/yolo26s-pose.pt).Hash -ine 'a083adb42303728ae14c4bd6bd56d80da46f82fb2564dbd6f31dcc92ea321646') { throw 'Weight SHA-256 mismatch' }
```

不匹配时不要运行模型。下载和校验完成后，可按[离线包 README](../README.md)离线运行。
