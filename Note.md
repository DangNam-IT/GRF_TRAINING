### Câu lệnh mở lại video train đã record
```
Get-ChildItem "experiments\videos" -Filter "*.avi" | Sort-Object LastWriteTime | ForEach-Object { Start-Process $_.FullName; Start-Sleep 2 }
```