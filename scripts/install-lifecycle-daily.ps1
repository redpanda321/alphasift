$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (!(Test-Path -LiteralPath $pythonExe)) { throw "Missing project Python: $pythonExe" }
& $pythonExe -m alphasift.lifecycle_daily --check
if ($LASTEXITCODE -ne 0) { throw 'Daily pipeline configuration check failed' }
$taskName = 'AlphaSift-Lifecycle-Daily'
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    throw 'Task already exists; inspect it before explicitly updating it.'
}
$action = New-ScheduledTaskAction -Execute $pythonExe -Argument '-m alphasift.lifecycle_daily' -WorkingDirectory $projectRoot
$trigger = New-ScheduledTaskTrigger -Daily -At '07:00'
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 18) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal -UserId $identity -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description 'Daily CN/US lifecycle research: daily+weekly, 5y/full-history, financial sector Top5. No monthly confirmation; no trading. Local time 07:00.'
Get-ScheduledTaskInfo -TaskName $taskName
