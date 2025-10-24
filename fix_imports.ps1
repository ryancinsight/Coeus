# Fix CpuBackend generic parameters throughout the codebase
# This script adds missing <T> generic parameters to CpuBackend types
# Required for proper B<S<T>> generic architecture compliance

Get-ChildItem -Recurse -Include "*.rs" -Exclude "*target*" | ForEach-Object {
    $filePath = $_.FullName
    try {
        $content = Get-Content $filePath -Raw -Encoding UTF8
        $originalContent = $content

        # Fix Tensor<CpuBackend, DenseStorage<T>, T> patterns
        $content = $content -replace 'Tensor<CpuBackend,\s*DenseStorage<([^>]+)>,\s*\1>', 'Tensor<CpuBackend<$1>, DenseStorage<$1>, $1>'

        # Fix standalone CpuBackend references in type annotations (when followed by comma or closing paren)
        $content = $content -replace 'CpuBackend,\s*DenseStorage<([^>]+)>,\s*\1', 'CpuBackend<$1>, DenseStorage<$1>, $1'

        # Fix function parameter type annotations
        $content = $content -replace 'Tensor<CpuBackend,\s*DenseStorage<([^>]+)>,\s*\1>', 'Tensor<CpuBackend<$1>, DenseStorage<$1>, $1>'

        # Fix return type annotations
        $content = $content -replace '\) ->.*Tensor<CpuBackend,\s*DenseStorage<([^>]+)>,\s*\1>', ') -> Tensor<CpuBackend<$1>, DenseStorage<$1>, $1>'

        # Fix remaining standalone CpuBackend without generics in type contexts
        # This is more complex - need to be careful about when to add <T>
        # For now, focus on the clear Tensor type patterns

        if ($content -ne $originalContent) {
            $content | Out-File -FilePath $filePath -Encoding UTF8 -NoNewline
            Write-Host "Fixed CpuBackend generics: $($_.Name)"
        }
    } catch {
        Write-Host "Error processing $($filePath): $($_.Exception.Message)"
    }
}
