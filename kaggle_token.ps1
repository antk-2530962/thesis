# 1. Create the .kaggle directory if it doesn't exist
New-Item -ItemType Directory -Force -Path "$HOME\.kaggle"

# 2. Write the access token to the file
"KGAT_24e1118b1b8cb3ceb277384f1f498214" | Out-File -FilePath "$HOME\.kaggle\access_token" -Encoding ascii

# 3. Restrict permissions (equivalent to chmod 600)
$path = "$HOME\.kaggle\access_token"
$acl = Get-Acl $path
$acl.SetAccessRuleProtection($true, $false) # Removes inherited permissions
$rule = New-Object System.Security.AccessControl.FileSystemAccessRule($env:USERNAME, "FullControl", "Allow")
$acl.SetAccessRule($rule)
Set-Acl $path $acl