<#
.SYNOPSIS
  Collects environment variables for SELO AI on Windows and writes them to backend/.env.

.DESCRIPTION
  Interactively prompts for BRAVE_SEARCH_API_KEY (masked input) and creates/updates
  the `backend/.env` file in the same directory as this script. Automatically generates
  SELO_SYSTEM_API_KEY and detects GPU tier for performance optimization.

.NOTES
  Run PowerShell with appropriate permissions. This script only modifies files within
  the current project directory.
#>

param(
  [string]$ProjectRoot
)

function Resolve-ProjectRoot {
  param([string]$Root)
  if ([string]::IsNullOrWhiteSpace($Root)) {
    return (Split-Path -Parent $MyInvocation.MyCommand.Path)
  }
  return (Resolve-Path $Root).Path
}

function New-RandomApiKey {
  param([int]$Length = 64)
  $bytes = New-Object byte[] ($Length / 2)
  $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
  $rng.GetBytes($bytes)
  return [System.BitConverter]::ToString($bytes).Replace('-', '').ToLower()
}

$RootDir = Resolve-ProjectRoot -Root $ProjectRoot
$BackendDir = Join-Path $RootDir 'backend'
$EnvPath = Join-Path $BackendDir '.env'

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "    SELO AI Windows Env Collector" -ForegroundColor Cyan
Write-Host "=========================================" 
Write-Host "Project root: $RootDir"
Write-Host "Env file:     $EnvPath"
Write-Host ""

# Ensure backend directory exists
if (-not (Test-Path $BackendDir)) {
  New-Item -ItemType Directory -Path $BackendDir -Force | Out-Null
}

# Load existing .env lines if present
$existing = @()
if (Test-Path $EnvPath) {
  $existing = Get-Content -Path $EnvPath -ErrorAction SilentlyContinue
}

# Helper: upsert key=value into .env content array
function Upsert-EnvLine {
  param(
    [Parameter(Mandatory=$true)][string]$Key,
    [Parameter(Mandatory=$true)][string]$Value,
    [Parameter(Mandatory=$true)][ref]$Lines
  )
  $pattern = "^$([regex]::Escape($Key))="
  $found = $false
  for ($i = 0; $i -lt $Lines.Value.Count; $i++) {
    if ($Lines.Value[$i] -match $pattern) {
      $Lines.Value[$i] = "$Key=$Value"
      $found = $true
      break
    }
  }
  if (-not $found) {
    $Lines.Value += "$Key=$Value"
  }
}

# Detect GPU tier and get configuration
Write-Host "Detecting GPU and performance tier..." -ForegroundColor Yellow
$tierScriptPath = Join-Path $RootDir "windows-detect-gpu-tier.ps1"
if (Test-Path $tierScriptPath) {
  . $tierScriptPath
  $tierConfig = Get-PerformanceTier
} else {
  Write-Host "Warning: GPU detection script not found. Using standard tier defaults." -ForegroundColor Yellow
  $tierConfig = @{
    Tier = "standard"
    ReflectionNumPredict = 640
    ReflectionMaxTokens = 640
    ReflectionWordMax = 500
    AnalyticalNumPredict = 640
    ChatNumPredict = 1024
    ChatNumCtx = 8192
  }
}

# Prompt for Brave Search API key (masked input)
$braveKey = $env:BRAVE_SEARCH_API_KEY
if (-not $braveKey) {
  Write-Host "Brave Search API key is required for web search features." -ForegroundColor Yellow
  Write-Host "Register for an API key at: https://search.brave.com/api"
  $secure = Read-Host -AsSecureString "Enter BRAVE_SEARCH_API_KEY (input hidden, press Enter to skip)"
  $ptr = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
  try {
    $braveKey = [System.Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)
  } finally {
    [System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)
  }
}

if (-not $braveKey) {
  Write-Host "No key entered. You can set BRAVE_SEARCH_API_KEY later in backend/.env." -ForegroundColor Yellow
  $braveKey = ""
}

# Generate SELO_SYSTEM_API_KEY automatically
$systemApiKey = New-RandomApiKey
Write-Host "Generated SELO_SYSTEM_API_KEY (ends with ****$($systemApiKey.Substring($systemApiKey.Length - 4)))" -ForegroundColor Green

# Prepare lines and write file
$lines = @($existing)

# Core API keys
Upsert-EnvLine -Key 'BRAVE_SEARCH_API_KEY' -Value $braveKey -Lines ([ref]$lines)
Upsert-EnvLine -Key 'SELO_SYSTEM_API_KEY' -Value $systemApiKey -Lines ([ref]$lines)

# Performance tier
Upsert-EnvLine -Key 'PERFORMANCE_TIER' -Value $tierConfig.Tier -Lines ([ref]$lines)

# Reflection configuration (tier-based)
Upsert-EnvLine -Key 'REFLECTION_NUM_PREDICT' -Value $tierConfig.ReflectionNumPredict -Lines ([ref]$lines)
Upsert-EnvLine -Key 'REFLECTION_MAX_TOKENS' -Value $tierConfig.ReflectionMaxTokens -Lines ([ref]$lines)
Upsert-EnvLine -Key 'REFLECTION_WORD_MAX' -Value $tierConfig.ReflectionWordMax -Lines ([ref]$lines)
Upsert-EnvLine -Key 'REFLECTION_WORD_MIN' -Value '170' -Lines ([ref]$lines)
Upsert-EnvLine -Key 'REFLECTION_TEMPERATURE' -Value '0.35' -Lines ([ref]$lines)

# Analytical configuration (tier-based)
Upsert-EnvLine -Key 'ANALYTICAL_NUM_PREDICT' -Value $tierConfig.AnalyticalNumPredict -Lines ([ref]$lines)
Upsert-EnvLine -Key 'ANALYTICAL_TEMPERATURE' -Value '0.2' -Lines ([ref]$lines)

# Chat configuration (tier-based)
Upsert-EnvLine -Key 'CHAT_NUM_PREDICT' -Value $tierConfig.ChatNumPredict -Lines ([ref]$lines)
Upsert-EnvLine -Key 'CHAT_NUM_CTX' -Value $tierConfig.ChatNumCtx -Lines ([ref]$lines)
Upsert-EnvLine -Key 'CHAT_TEMPERATURE' -Value '0.6' -Lines ([ref]$lines)

# Selective reflection classifier (metacognitive system)
Upsert-EnvLine -Key 'REFLECTION_CLASSIFIER_ENABLED' -Value 'true' -Lines ([ref]$lines)
Upsert-EnvLine -Key 'REFLECTION_CLASSIFIER_MODEL' -Value 'same' -Lines ([ref]$lines)
Upsert-EnvLine -Key 'REFLECTION_CLASSIFIER_THRESHOLD' -Value 'balanced' -Lines ([ref]$lines)
Upsert-EnvLine -Key 'REFLECTION_MANDATORY_INTERVAL' -Value '10' -Lines ([ref]$lines)
Upsert-EnvLine -Key 'REFLECTION_MANDATORY_EARLY_TURNS' -Value '5' -Lines ([ref]$lines)

# Ensure essential defaults if missing
function Ensure-Default {
  param([string]$Key, [string]$Value)
  $pattern = "^$([regex]::Escape($Key))="
  if (-not ($lines | Select-String -Pattern $pattern -SimpleMatch)) {
    $lines += "$Key=$Value"
  }
}

Ensure-Default -Key 'HOST' -Value '0.0.0.0'
Ensure-Default -Key 'PORT' -Value '8000'
Ensure-Default -Key 'CORS_ORIGINS' -Value 'http://localhost:3000'
Ensure-Default -Key 'DATABASE_URL' -Value 'sqlite+aiosqlite:///selo_ai_beta.db'

# Write file
$lines | Set-Content -Path $EnvPath -Encoding UTF8

# Confirm
Write-Host "" 
Write-Host "=========================================" -ForegroundColor Green
Write-Host "    Configuration Complete!" -ForegroundColor Green
Write-Host "=========================================" 
Write-Host "Performance Tier: $($tierConfig.Tier)" -ForegroundColor Cyan
Write-Host "Reflection Tokens: $($tierConfig.ReflectionMaxTokens)" -ForegroundColor Cyan
Write-Host "Chat Context: $($tierConfig.ChatNumCtx)" -ForegroundColor Cyan
Write-Host "" 
$tail = if ($braveKey.Length -gt 4) { $braveKey.Substring($braveKey.Length-4) } else { "none" }
if ($braveKey) {
  Write-Host "✅ Saved BRAVE_SEARCH_API_KEY (ends with ****$tail)" -ForegroundColor Green
} else {
  Write-Host "⚠️  No BRAVE_SEARCH_API_KEY set (web search disabled)" -ForegroundColor Yellow
}
Write-Host "✅ Generated SELO_SYSTEM_API_KEY" -ForegroundColor Green
Write-Host "✅ Configured tier-based token budgets" -ForegroundColor Green
Write-Host "✅ Enabled selective reflection classifier (50-65% faster)" -ForegroundColor Green
Write-Host "" 
Write-Host "Configuration saved to: $EnvPath" -ForegroundColor White
