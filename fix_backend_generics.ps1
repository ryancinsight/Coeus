# PowerShell script to fix CpuBackend generic parameters
param(
    [string]$Path = "."
)

# Find all Rust files
$rustFiles = Get-ChildItem -Path $Path -Recurse -Include "*.rs" -File

foreach ($file in $rustFiles) {
    $content = Get-Content -Path $file.FullName -Raw

    # Pattern: CpuBackend followed by whitespace or punctuation, not followed by '<'
    # We want to replace CpuBackend followed by anything except '<' with CpuBackend<T>
    $pattern = '(?<!<)CpuBackend(?![<])'
    $replacement = 'CpuBackend<T>'

    if ($content -match $pattern) {
        Write-Host "Fixing $file"
        $newContent = $content -replace $pattern, $replacement
        Set-Content -Path $file.FullName -Value $newContent
    }
}

Write-Host "Backend generics fix complete"
