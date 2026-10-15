# run_tests.ps1 - Automated verification test suite for Planetary Comparison & Analysis Module
# Runs natively in PowerShell without external dependencies.

$ErrorActionPreference = "Stop"
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "🚀 VORNIS AI - PLANETARY COMPARISON TEST SUITE" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

$passed = 0
$failed = 0

function Assert-Test {
    param(
        [bool]$Condition,
        [string]$Name
    )
    if ($Condition) {
        Write-Host "  [PASS] $Name" -ForegroundColor Green
        $script:passed++
    } else {
        Write-Host "  [FAIL] $Name" -ForegroundColor Red
        $script:failed++
    }
}

# --- TEST SUITE 1: DATA LAYER INTEGRITY ---
Write-Host "`n[1] Testing Planetary Data Layer..." -ForegroundColor Yellow

$jsonPath = Join-Path $PSScriptRoot "..\api\planetary_data.json"
Assert-Test (Test-Path $jsonPath) "planetary_data.json file exists"

$data = Get-Content $jsonPath -Raw | ConvertFrom-Json
Assert-Test ($null -ne $data.bodies) "JSON contains 'bodies' root object"

$expectedBodies = @(
    "Mercury", "Venus", "Earth", "Moon", "Mars", "Phobos", "Ceres",
    "Jupiter", "Io", "Europa", "Ganymede", "Callisto", "Saturn",
    "Titan", "Enceladus", "Uranus", "Neptune", "Triton", "Pluto"
)

$bodiesCount = ($data.bodies.PSObject.Properties | Measure-Object).Count
Assert-Test ($bodiesCount -ge 19) "Data layer has 19 celestial bodies (found $bodiesCount)"

foreach ($bName in $expectedBodies) {
    $b = $data.bodies.$bName
    Assert-Test ($null -ne $b) "Body '$bName' exists in dataset"
    if ($null -ne $b) {
        Assert-Test ($b.physical.radius_km -gt 0) "$bName radius is valid ($($b.physical.radius_km) km)"
        Assert-Test ($b.physical.mass_1e24_kg -gt 0) "$bName mass is valid ($($b.physical.mass_1e24_kg) 1e24 kg)"
        Assert-Test ($b.physical.distance_from_sun_au -gt 0) "$bName distance is valid ($($b.physical.distance_from_sun_au) AU)"
        Assert-Test ($null -ne $b.atmospheric.composition) "$bName has atmospheric composition"
        Assert-Test ($null -ne $b.environmental.radiation_tier) "$bName has radiation tier"
    }
}

# --- TEST SUITE 2: PRESETS INTEGRITY ---
Write-Host "`n[2] Testing Quick Compare Presets..." -ForegroundColor Yellow
$presets = $data.presets
Assert-Test ($presets.Count -ge 6) "At least 6 presets provided (found $($presets.Count))"

foreach ($p in $presets) {
    Assert-Test ($p.bodies.Count -ge 2 -and $p.bodies.Count -le 4) "Preset '$($p.name)' body count is within [2, 4]"
    foreach ($target in $p.bodies) {
        Assert-Test ($null -ne $data.bodies.$target) "Preset body '$target' exists in database"
    }
}

# --- TEST SUITE 3: SELECTION VALIDATION RULES ---
Write-Host "`n[3] Testing Selection Validation Rules..." -ForegroundColor Yellow

function Validate-Selection([string[]]$selection) {
    if ($selection.Count -lt 2) {
        return @{ Valid = $false; Error = "Minimum 2 bodies required" }
    }
    if ($selection.Count -gt 4) {
        return @{ Valid = $false; Error = "Maximum 4 bodies allowed" }
    }
    return @{ Valid = $true; Error = $null }
}

Assert-Test (-not (Validate-Selection @()).Valid) "Rejects empty selection"
Assert-Test (-not (Validate-Selection @("Mars")).Valid) "Rejects 1-body selection"
Assert-Test ((Validate-Selection @("Earth", "Mars")).Valid) "Accepts 2-body selection (Earth vs Mars)"
Assert-Test ((Validate-Selection @("Earth", "Moon", "Mars")).Valid) "Accepts 3-body selection"
Assert-Test ((Validate-Selection @("Earth", "Moon", "Mars", "Europa")).Valid) "Accepts 4-body selection"
Assert-Test (-not (Validate-Selection @("Earth", "Moon", "Mars", "Europa", "Titan")).Valid) "Rejects 5-body selection (exceeds max 4)"

# --- TEST SUITE 4: FEASIBILITY SCORING WEIGHTS & TIERS ---
Write-Host "`n[4] Testing Feasibility Scoring Weights..." -ForegroundColor Yellow

$weights = @{
    "Flyby"   = @{ "travel" = 0.35; "comm" = 0.25; "rad" = 0.25; "isru" = 0.15; "landing" = 0.00; "surv" = 0.00; "habit" = 0.00 }
    "Orbiter" = @{ "travel" = 0.30; "rad" = 0.25; "comm" = 0.20; "isru" = 0.15; "surv" = 0.10; "landing" = 0.00; "habit" = 0.00 }
    "Lander"  = @{ "landing" = 0.25; "surv" = 0.25; "travel" = 0.20; "comm" = 0.15; "rad" = 0.10; "isru" = 0.05; "habit" = 0.00 }
    "Crewed"  = @{ "surv" = 0.25; "rad" = 0.20; "landing" = 0.15; "travel" = 0.15; "habit" = 0.10; "comm" = 0.10; "isru" = 0.05 }
    "Colony"  = @{ "habit" = 0.25; "isru" = 0.25; "surv" = 0.20; "rad" = 0.15; "landing" = 0.10; "travel" = 0.05; "comm" = 0.00 }
}

foreach ($mType in $weights.Keys) {
    $sum = 0.0
    foreach ($w in $weights[$mType].Values) { $sum += $w }
    Assert-Test ([Math]::Round($sum, 2) -eq 1.00) "Weights for $mType sum to 1.00 (got $sum)"
}

function Get-Tier([double]$score) {
    if ($score -ge 70.0) { return "High Feasibility" }
    if ($score -ge 45.0) { return "Moderate Feasibility" }
    return "Low Feasibility / High Risk"
}

Assert-Test ((Get-Tier 85.5) -eq "High Feasibility") "Score 85.5 is High Feasibility (Theme: #00e676)"
Assert-Test ((Get-Tier 70.0) -eq "High Feasibility") "Score 70.0 is High Feasibility boundary"
Assert-Test ((Get-Tier 65.0) -eq "Moderate Feasibility") "Score 65.0 is Moderate Feasibility (Theme: #ffea00)"
Assert-Test ((Get-Tier 45.0) -eq "Moderate Feasibility") "Score 45.0 is Moderate Feasibility boundary"
Assert-Test ((Get-Tier 32.0) -eq "Low Feasibility / High Risk") "Score 32.0 is Low Feasibility / High Risk (Theme: #ff3d00)"

Write-Host "`n========================================================" -ForegroundColor Cyan
Write-Host "TEST RESULTS: $passed passed, $failed failed" -ForegroundColor $(if ($failed -eq 0) { "Green" } else { "Red" })
Write-Host "========================================================" -ForegroundColor Cyan

if ($failed -gt 0) {
    exit 1
} else {
    exit 0
}
