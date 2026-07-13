Get-CimInstance -ClassName Win32_LogicalDisk -Filter "DriveType=3" | Select-Object `
    @{Name="Drive";Expression={$_.DeviceID}},
    @{Name="Total (GB)";Expression={[Math]::Round($_.Size / 1GB, 2)}},
    @{Name="Free (GB)";Expression={[Math]::Round($_.FreeSpace / 1GB, 2)}},
    @{Name="Used (GB)";Expression={[Math]::Round(($_.Size - $_.FreeSpace) / 1GB, 2)}},
    @{Name="Free %";Expression={[Math]::Round(($_.FreeSpace / $_.Size) * 100, 2)}} |
    Format-Table -AutoSize
