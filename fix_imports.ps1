# PowerShell script to fix import inconsistencies in examples

$files = Get-ChildItem -Path "examples" -Filter "*.rs" -Recurse

foreach ($file in $files) {
    $content = Get-Content $file.FullName -Raw

    # Fix all the incorrect import prefixes
    $content = $content -replace 'coeus_nn::', 'nn::'
    $content = $content -replace 'coeus_tensor::', 'tensor::'
    $content = $content -replace 'coeus_autograd::', 'autograd::'
    $content = $content -replace 'coeus_backend::', 'backend::'
    $content = $content -replace 'coeus_foundation::', 'foundation::'

    # Fix direct crate references too
    $content = $content -replace 'coeus_nn', 'nn'
    $content = $content -replace 'coeus_tensor', 'tensor'
    $content = $content -replace 'coeus_autograd', 'autograd'
    $content = $content -replace 'coeus_backend', 'backend'
    $content = $content -replace 'coeus_foundation', 'foundation'

    Set-Content $file.FullName $content
}

Write-Host "Import fixes completed for all example files"