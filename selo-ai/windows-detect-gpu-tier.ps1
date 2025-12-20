<#
.SYNOPSIS
  Detect GPU tier for SELO AI Windows installation.

.DESCRIPTION
  Detects GPU VRAM and determines performance tier (standard/high).
  Returns tier-specific token budget configuration.

.OUTPUTS
  Hashtable with tier and configuration values.

.NOTES
  Supports NVIDIA GPUs via nvidia-smi.
  Falls back to standard tier if GPU not detected.
#>

function Get-GpuMemory {
    param()
    
    $gpuMemoryMB = 0
    $gpuName = "Unknown"
    
    # Try nvidia-smi first (most reliable)
    try {
        $nvidiaSmiPath = Get-Command nvidia-smi -ErrorAction SilentlyContinue
        if ($nvidiaSmiPath) {
            $output = & nvidia-smi --query-gpu=memory.total,name --format=csv,noheader,nounits 2>$null
            if ($output) {
                $parts = $output.Split(',')
                if ($parts.Length -ge 1) {
                    $gpuMemoryMB = [int]$parts[0].Trim()
                }
                if ($parts.Length -ge 2) {
                    $gpuName = $parts[1].Trim()
                }
                Write-Host "GPU detected via nvidia-smi: $gpuName ($gpuMemoryMB MB)" -ForegroundColor Cyan
                return @{
                    MemoryMB = $gpuMemoryMB
                    Name = $gpuName
                    Method = "nvidia-smi"
                }
            }
        }
    } catch {
        Write-Verbose "nvidia-smi detection failed: $($_.Exception.Message)"
    }
    
    # Try WMI as fallback (less reliable for VRAM)
    try {
        $gpu = Get-WmiObject Win32_VideoController -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($gpu) {
            $gpuName = $gpu.Name
            # AdapterRAM is in bytes, convert to MB
            if ($gpu.AdapterRAM -and $gpu.AdapterRAM -gt 0) {
                $gpuMemoryMB = [math]::Round($gpu.AdapterRAM / 1MB)
                Write-Host "GPU detected via WMI: $gpuName ($gpuMemoryMB MB)" -ForegroundColor Cyan
                return @{
                    MemoryMB = $gpuMemoryMB
                    Name = $gpuName
                    Method = "WMI"
                }
            }
        }
    } catch {
        Write-Verbose "WMI detection failed: $($_.Exception.Message)"
    }
    
    Write-Host "No GPU detected or GPU tools unavailable - using standard tier" -ForegroundColor Yellow
    return @{
        MemoryMB = 0
        Name = "None"
        Method = "none"
    }
}

function Get-PerformanceTier {
    param()
    
    $gpuInfo = Get-GpuMemory
    $gpuMemoryMB = $gpuInfo.MemoryMB
    $gpuName = $gpuInfo.Name
    
    # Determine tier: high = 12GB+ (12288MB), standard = <12GB
    if ($gpuMemoryMB -ge 12288) {
        $tier = "high"
        Write-Host "🎯 High-Performance Tier Activated" -ForegroundColor Green
        Write-Host "   GPU: $gpuName (${gpuMemoryMB}MB VRAM)" -ForegroundColor Green
        Write-Host "   - Reflection capacity: 650 tokens (~650 words max)" -ForegroundColor White
        Write-Host "   - Enhanced philosophical depth during persona bootstrap" -ForegroundColor White
        Write-Host "   - Extended chat context: 8192 tokens" -ForegroundColor White
    } else {
        $tier = "standard"
        Write-Host "⚡ Standard Tier Activated (optimized for 8GB GPU)" -ForegroundColor Green
        if ($gpuMemoryMB -gt 0) {
            Write-Host "   GPU: $gpuName (${gpuMemoryMB}MB VRAM)" -ForegroundColor Green
        }
        Write-Host "   - Reflection capacity: 640 tokens (~500 words max)" -ForegroundColor White
        Write-Host "   - Context window: 8192 tokens (qwen2.5:3b native capacity)" -ForegroundColor White
        Write-Host "   - Full-quality few-shot examples preserved" -ForegroundColor White
    }
    Write-Host ""
    
    # Return tier configuration
    $config = @{
        Tier = $tier
        GpuMemoryMB = $gpuMemoryMB
        GpuName = $gpuName
        DetectionMethod = $gpuInfo.Method
    }
    
    # Add tier-specific token budgets
    if ($tier -eq "high") {
        $config.ReflectionNumPredict = 650
        $config.ReflectionMaxTokens = 650
        $config.ReflectionWordMax = 650
        $config.AnalyticalNumPredict = 1536
        $config.ChatNumPredict = 2048
        $config.ChatNumCtx = 8192
    } else {
        $config.ReflectionNumPredict = 640
        $config.ReflectionMaxTokens = 640
        $config.ReflectionWordMax = 500
        $config.AnalyticalNumPredict = 640
        $config.ChatNumPredict = 1024
        $config.ChatNumCtx = 8192
    }
    
    return $config
}

function Export-TierConfiguration {
    param(
        [Parameter(Mandatory=$true)]
        [hashtable]$Config,
        
        [Parameter(Mandatory=$true)]
        [string]$OutputPath
    )
    
    $envLines = @()
    
    # Tier-based token budgets
    $envLines += "# Performance Tier: $($Config.Tier)"
    $envLines += "# GPU: $($Config.GpuName) ($($Config.GpuMemoryMB)MB)"
    $envLines += "PERFORMANCE_TIER=$($Config.Tier)"
    $envLines += ""
    
    # Reflection configuration
    $envLines += "# Reflection Configuration"
    $envLines += "REFLECTION_NUM_PREDICT=$($Config.ReflectionNumPredict)"
    $envLines += "REFLECTION_MAX_TOKENS=$($Config.ReflectionMaxTokens)"
    $envLines += "REFLECTION_WORD_MAX=$($Config.ReflectionWordMax)"
    $envLines += "REFLECTION_WORD_MIN=170"
    $envLines += "REFLECTION_TEMPERATURE=0.35"
    $envLines += ""
    
    # Analytical configuration
    $envLines += "# Analytical Configuration"
    $envLines += "ANALYTICAL_NUM_PREDICT=$($Config.AnalyticalNumPredict)"
    $envLines += "ANALYTICAL_TEMPERATURE=0.2"
    $envLines += ""
    
    # Chat configuration
    $envLines += "# Chat Configuration"
    $envLines += "CHAT_NUM_PREDICT=$($Config.ChatNumPredict)"
    $envLines += "CHAT_NUM_CTX=$($Config.ChatNumCtx)"
    $envLines += "CHAT_TEMPERATURE=0.6"
    $envLines += ""
    
    # Selective reflection classifier
    $envLines += "# Selective Reflection Classifier (Metacognitive System)"
    $envLines += "REFLECTION_CLASSIFIER_ENABLED=true"
    $envLines += "REFLECTION_CLASSIFIER_MODEL=same"
    $envLines += "REFLECTION_CLASSIFIER_THRESHOLD=balanced"
    $envLines += "REFLECTION_MANDATORY_INTERVAL=10"
    $envLines += "REFLECTION_MANDATORY_EARLY_TURNS=5"
    $envLines += ""
    
    # Write to file
    $envLines | Out-File -FilePath $OutputPath -Encoding UTF8 -Append
    
    Write-Host "✅ Tier configuration written to $OutputPath" -ForegroundColor Green
}

# Export functions for module usage (only when loaded as a module)
if ($MyInvocation.Line -match '^\s*Import-Module') {
    Export-ModuleMember -Function Get-GpuMemory, Get-PerformanceTier, Export-TierConfiguration
}
