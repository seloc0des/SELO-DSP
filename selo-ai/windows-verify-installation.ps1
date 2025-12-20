<#
.SYNOPSIS
  Verify SELO AI Windows installation configuration.

.DESCRIPTION
  Checks that all required configuration is present and correct after installation.
  Validates GPU detection, tier configuration, and environment variables.

.NOTES
  Run after installation to verify everything is configured correctly.
#>

param(
    [string]$ProjectRoot = "."
)

$ErrorCount = 0
$WarningCount = 0

function Write-Check {
    param(
        [string]$Message,
        [string]$Status,
        [string]$Detail = ""
    )
    
    $symbol = switch ($Status) {
        "pass" { "✅" }
        "fail" { "❌" }
        "warn" { "⚠️" }
        default { "ℹ️" }
    }
    
    $color = switch ($Status) {
        "pass" { "Green" }
        "fail" { "Red" }
        "warn" { "Yellow" }
        default { "White" }
    }
    
    Write-Host "$symbol $Message" -ForegroundColor $color
    if ($Detail) {
        Write-Host "   $Detail" -ForegroundColor Gray
    }
}

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  SELO AI Installation Verification" -ForegroundColor Cyan
Write-Host "========================================="
Write-Host ""

# Check 1: Backend directory exists
Write-Host "Checking installation structure..." -ForegroundColor Yellow
$backendDir = Join-Path $ProjectRoot "backend"
if (Test-Path $backendDir) {
    Write-Check "Backend directory exists" "pass"
} else {
    Write-Check "Backend directory missing" "fail" "Expected: $backendDir"
    $ErrorCount++
}

# Check 2: .env file exists
$envPath = Join-Path $backendDir ".env"
if (Test-Path $envPath) {
    Write-Check ".env file exists" "pass"
} else {
    Write-Check ".env file missing" "fail" "Expected: $envPath"
    $ErrorCount++
    Write-Host ""
    Write-Host "CRITICAL: Cannot continue without .env file" -ForegroundColor Red
    exit 1
}

# Load .env file
$envContent = Get-Content $envPath -ErrorAction SilentlyContinue
$envVars = @{}
foreach ($line in $envContent) {
    if ($line -match '^([^=]+)=(.*)$') {
        $envVars[$matches[1]] = $matches[2]
    }
}

Write-Host ""
Write-Host "Checking GPU and tier configuration..." -ForegroundColor Yellow

# Check 3: Performance tier
if ($envVars.ContainsKey("PERFORMANCE_TIER")) {
    $tier = $envVars["PERFORMANCE_TIER"]
    if ($tier -eq "high" -or $tier -eq "standard") {
        Write-Check "Performance tier detected: $tier" "pass"
    } else {
        Write-Check "Invalid performance tier: $tier" "fail" "Expected: 'high' or 'standard'"
        $ErrorCount++
    }
} else {
    Write-Check "PERFORMANCE_TIER not set" "warn" "Will use runtime defaults"
    $WarningCount++
}

# Check 4: GPU detection
Write-Host ""
Write-Host "Checking GPU availability..." -ForegroundColor Yellow
try {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidiaSmi) {
        $gpuInfo = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null
        if ($gpuInfo) {
            Write-Check "NVIDIA GPU detected" "pass" $gpuInfo
        } else {
            Write-Check "nvidia-smi found but no GPU info" "warn"
            $WarningCount++
        }
    } else {
        Write-Check "nvidia-smi not found" "info" "CPU mode or non-NVIDIA GPU"
    }
} catch {
    Write-Check "GPU detection failed" "warn" $_.Exception.Message
    $WarningCount++
}

Write-Host ""
Write-Host "Checking reflection configuration..." -ForegroundColor Yellow

# Check 5: Reflection settings
$requiredReflectionVars = @(
    "REFLECTION_NUM_PREDICT",
    "REFLECTION_MAX_TOKENS",
    "REFLECTION_WORD_MAX",
    "REFLECTION_WORD_MIN",
    "REFLECTION_TEMPERATURE"
)

foreach ($var in $requiredReflectionVars) {
    if ($envVars.ContainsKey($var)) {
        $value = $envVars[$var]
        Write-Check "$var = $value" "pass"
    } else {
        Write-Check "$var not set" "warn" "Will use runtime defaults"
        $WarningCount++
    }
}

Write-Host ""
Write-Host "Checking chat configuration..." -ForegroundColor Yellow

# Check 6: Chat settings
$requiredChatVars = @(
    "CHAT_NUM_PREDICT",
    "CHAT_NUM_CTX",
    "CHAT_TEMPERATURE"
)

foreach ($var in $requiredChatVars) {
    if ($envVars.ContainsKey($var)) {
        $value = $envVars[$var]
        Write-Check "$var = $value" "pass"
    } else {
        Write-Check "$var not set" "warn" "Will use runtime defaults"
        $WarningCount++
    }
}

Write-Host ""
Write-Host "Checking selective reflection..." -ForegroundColor Yellow

# Check 7: Selective reflection classifier
if ($envVars.ContainsKey("REFLECTION_CLASSIFIER_ENABLED")) {
    $enabled = $envVars["REFLECTION_CLASSIFIER_ENABLED"]
    if ($enabled -eq "true") {
        Write-Check "Selective reflection enabled" "pass" "50-65% faster responses"
    } else {
        Write-Check "Selective reflection disabled" "warn" "Responses will be slower"
        $WarningCount++
    }
} else {
    Write-Check "REFLECTION_CLASSIFIER_ENABLED not set" "warn" "Will use runtime defaults"
    $WarningCount++
}

Write-Host ""
Write-Host "Checking API keys..." -ForegroundColor Yellow

# Check 8: System API key
if ($envVars.ContainsKey("SELO_SYSTEM_API_KEY")) {
    $key = $envVars["SELO_SYSTEM_API_KEY"]
    if ($key.Length -ge 32) {
        Write-Check "SELO_SYSTEM_API_KEY present" "pass" "Length: $($key.Length) chars"
    } else {
        Write-Check "SELO_SYSTEM_API_KEY too short" "fail" "Expected: 32+ chars, Got: $($key.Length)"
        $ErrorCount++
    }
} else {
    Write-Check "SELO_SYSTEM_API_KEY not set" "fail" "Required for system operation"
    $ErrorCount++
}

# Check 9: Brave Search API key
if ($envVars.ContainsKey("BRAVE_SEARCH_API_KEY")) {
    $key = $envVars["BRAVE_SEARCH_API_KEY"]
    if ($key.Length -gt 0) {
        Write-Check "BRAVE_SEARCH_API_KEY present" "pass" "Web search enabled"
    } else {
        Write-Check "BRAVE_SEARCH_API_KEY empty" "info" "Web search disabled"
    }
} else {
    Write-Check "BRAVE_SEARCH_API_KEY not set" "info" "Web search disabled"
}

Write-Host ""
Write-Host "Checking database configuration..." -ForegroundColor Yellow

# Check 10: Database URL
if ($envVars.ContainsKey("DATABASE_URL")) {
    $dbUrl = $envVars["DATABASE_URL"]
    if ($dbUrl -match "sqlite") {
        Write-Check "Database: SQLite" "pass" "Recommended for Windows"
    } elseif ($dbUrl -match "postgresql") {
        Write-Check "Database: PostgreSQL" "pass" "Advanced configuration"
    } else {
        Write-Check "Unknown database type" "warn" $dbUrl
        $WarningCount++
    }
} else {
    Write-Check "DATABASE_URL not set" "fail" "Required for operation"
    $ErrorCount++
}

Write-Host ""
Write-Host "Checking Ollama service..." -ForegroundColor Yellow

# Check 11: Ollama availability
try {
    $response = Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/tags" -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
    if ($response.StatusCode -eq 200) {
        Write-Check "Ollama service responding" "pass" "http://127.0.0.1:11434"
    } else {
        Write-Check "Ollama service returned status $($response.StatusCode)" "warn"
        $WarningCount++
    }
} catch {
    Write-Check "Ollama service not responding" "fail" "Ensure Ollama is running"
    $ErrorCount++
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  Verification Summary" -ForegroundColor Cyan
Write-Host "========================================="
Write-Host ""

if ($ErrorCount -eq 0 -and $WarningCount -eq 0) {
    Write-Host "✅ All checks passed!" -ForegroundColor Green
    Write-Host "   Installation is correctly configured." -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor White
    Write-Host "  1. Run start-beta.bat to start SELO AI" -ForegroundColor White
    Write-Host "  2. Wait for persona bootstrap (5-10 minutes first time)" -ForegroundColor White
    Write-Host "  3. Open http://localhost:3000 in your browser" -ForegroundColor White
} elseif ($ErrorCount -eq 0) {
    Write-Host "⚠️  $WarningCount warning(s) found" -ForegroundColor Yellow
    Write-Host "   Installation will work but may not be optimal." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Recommended action:" -ForegroundColor White
    Write-Host "  Re-run: .\windows-collect-env.ps1" -ForegroundColor White
} else {
    Write-Host "❌ $ErrorCount error(s) and $WarningCount warning(s) found" -ForegroundColor Red
    Write-Host "   Installation is incomplete or misconfigured." -ForegroundColor Red
    Write-Host ""
    Write-Host "Required action:" -ForegroundColor White
    Write-Host "  Re-run: .\windows-collect-env.ps1" -ForegroundColor White
    exit 1
}

Write-Host ""
