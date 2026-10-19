param([string]$ImagePath, [string]$Language='ko', [string]$BatchPath, [switch]$Server)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
[Console]::InputEncoding = New-Object System.Text.UTF8Encoding($false)
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Storage.StorageFile,Windows.Storage,ContentType=WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder,Windows.Graphics.Imaging,ContentType=WindowsRuntime]
$null = [Windows.Media.Ocr.OcrEngine,Windows.Foundation,ContentType=WindowsRuntime]
$null = [Windows.Globalization.Language,Windows.Globalization,ContentType=WindowsRuntime]
$null = [Windows.Storage.Streams.InMemoryRandomAccessStream,Windows.Storage.Streams,ContentType=WindowsRuntime]
$null = [Windows.Storage.Streams.DataWriter,Windows.Storage.Streams,ContentType=WindowsRuntime]
$script:awaitMethod = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' } | Select-Object -First 1
function Await($Operation, [Type]$ResultType) {
    $task = $script:awaitMethod.MakeGenericMethod($ResultType).Invoke($null, @($Operation))
    $task.GetAwaiter().GetResult()
}
$script:engines = @{}
function Read-Image([string]$Path, [string]$Lang) {
$stream = [Windows.Storage.Streams.InMemoryRandomAccessStream]::new()
$writer = [Windows.Storage.Streams.DataWriter]::new($stream)
try {
    $writer.WriteBytes([System.IO.File]::ReadAllBytes($Path))
    $null = Await ($writer.StoreAsync()) ([uint32])
    $null = $writer.DetachStream()
} finally { $writer.Dispose() }
$stream.Seek(0)
try {
    $decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
    $bitmap = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
    try {
        if (-not $script:engines.ContainsKey($Lang)) {
            $script:engines[$Lang] = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage([Windows.Globalization.Language]::new($Lang))
        }
        $engine = $script:engines[$Lang]
        if ($null -eq $engine) { throw 'Windows OCR language pack is unavailable.' }
        $result = Await ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
        $lines = @($result.Lines | ForEach-Object {
            $words = @($_.Words | ForEach-Object { @{text=$_.Text; x=$_.BoundingRect.X; y=$_.BoundingRect.Y; w=$_.BoundingRect.Width; h=$_.BoundingRect.Height} })
            @{text=$_.Text; words=$words}
        })
        @{text=$result.Text; lines=$lines; language=$engine.RecognizerLanguage.LanguageTag}
    } finally { if ($bitmap) { $bitmap.Dispose() } }
} finally { $stream.Dispose() }
}
if ($Server) {
    while ($null -ne ($line = [Console]::ReadLine())) {
        try {
            $requests = ConvertFrom-Json -InputObject $line
            $results = @(foreach ($request in $requests) { Read-Image $request.path $request.language })
            [Console]::WriteLine((ConvertTo-Json -InputObject $results -Depth 8 -Compress))
        } catch {
            [Console]::WriteLine((ConvertTo-Json -InputObject @{error=$_.Exception.Message} -Compress))
        }
    }
} elseif ($BatchPath) {
    $requests = Get-Content -LiteralPath $BatchPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $results = @(foreach ($request in $requests) { Read-Image $request.path $request.language })
    ConvertTo-Json -InputObject $results -Depth 8 -Compress
} else {
    Read-Image $ImagePath $Language | ConvertTo-Json -Depth 7 -Compress
}
