<#
.SYNOPSIS
  Configure Windows Defender Firewall rules for SELO AI.

.DESCRIPTION
  Prompts the user to add inbound TCP rules for the SELO AI backend (default 8000)
  and frontend (3000). If rules already exist, the script will skip creating duplicates.

.NOTES
  Run PowerShell as Administrator.
#>

param(
  [int]$BackendPort = 8000,
  [int]$FrontendPort = 3000
)

function Confirm-Action {
  param([string]$Message)
  $choice = Read-Host "$Message (y/N)"
  return ($choice -match '^(?i)y(es)?$')
}

function Ensure-FirewallRule {
  param(
    [Parameter(Mandatory=$true)][string]$DisplayName,
    [Parameter(Mandatory=$true)][int]$Port
  )
  $exists = Get-NetFirewallRule -DisplayName $DisplayName -ErrorAction SilentlyContinue | Where-Object { $_.Enabled -eq 'True' }
  if ($exists) {
    Write-Host "Rule '$DisplayName' already exists. Skipping." -ForegroundColor Yellow
    return
  }
  try {
    New-NetFirewallRule -DisplayName $DisplayName -Direction Inbound -Action Allow -Protocol TCP -LocalPort $Port | Out-Null
    Write-Host "Created firewall rule '$DisplayName' for TCP $Port" -ForegroundColor Green
  } catch {
    Write-Host "Failed to create firewall rule '$DisplayName': $($_.Exception.Message)" -ForegroundColor Red
  }
}

Write-Host "========================================="
Write-Host "    Windows Firewall Configuration" -ForegroundColor Cyan
Write-Host "========================================="
Write-Host "This script will create inbound allow rules for:" 
Write-Host "  • Backend (TCP $BackendPort)" 
Write-Host "  • Frontend (TCP $FrontendPort)"
Write-Host ""

if (-not (Confirm-Action "Proceed to add Windows Defender Firewall rules now?")) {
  Write-Host "Canceled by user. No changes made." -ForegroundColor Yellow
  exit 0
}

# Require elevation
$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
  Write-Host "This script must be run as Administrator. Right-click PowerShell and choose 'Run as administrator'." -ForegroundColor Red
  exit 1
}

Ensure-FirewallRule -DisplayName "SELO AI Backend ($BackendPort)" -Port $BackendPort
Ensure-FirewallRule -DisplayName "SELO AI Frontend ($FrontendPort)" -Port $FrontendPort

Write-Host "Done. You can manage rules in Windows Defender Firewall with Advanced Security." -ForegroundColor Green
