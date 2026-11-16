$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$file = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot 'Manual/01_Khung_bao_cao_RAG.docx'))
$out = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot 'qa/word_pages'))
[IO.Directory]::CreateDirectory($out) | Out-Null
$word = $null
$doc = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $true
    $word.DisplayAlerts = 0
    $word.AutomationSecurity = 3
    $word.Options.UpdateLinksAtOpen = $false
    $doc = $word.Documents.Open($file, $false, $false)
    $doc.ActiveWindow.View.Type = 3
    $doc.Repaginate()
    $doc.Fields.Update() | Out-Null
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    foreach ($sec in $doc.Sections) {
        foreach ($footer in $sec.Footers) { if ($footer.Exists) { $footer.Range.Fields.Update() | Out-Null } }
    }
    $doc.Repaginate()
    foreach ($toc in $doc.TablesOfContents) { $toc.UpdatePageNumbers() }
    $doc.Save()
    $doc.Repaginate()
    $pages = $doc.ActiveWindow.Panes.Item(1).Pages
    $count = $pages.Count
    Write-Output ('Pages=' + $count + ' TOCs=' + $doc.TablesOfContents.Count)
    for ($i=1; $i -le $count; $i++) {
        $page = $pages.Item($i)
        [byte[]]$bits = $page.EnhMetaFileBits
        $stream = New-Object IO.MemoryStream(,$bits)
        $meta = New-Object Drawing.Imaging.Metafile($stream)
        $bitmap = New-Object Drawing.Bitmap([int]($page.Width*2), [int]($page.Height*2))
        $graphics = [Drawing.Graphics]::FromImage($bitmap)
        $graphics.Clear([Drawing.Color]::White)
        $graphics.DrawImage($meta, 0, 0, $bitmap.Width, $bitmap.Height)
        $bitmap.Save((Join-Path $out ('page-' + $i + '.png')), [Drawing.Imaging.ImageFormat]::Png)
        $graphics.Dispose(); $bitmap.Dispose(); $meta.Dispose(); $stream.Dispose()
        Write-Output ('Rendered page ' + $i)
    }
    $info = [ordered]@{pages=$count; tables_of_contents=$doc.TablesOfContents.Count; word_version=$word.Version; renderer='Word native page EMF to PNG'; pdf_created=$false}
    $info | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $PSScriptRoot 'qa/render_info.json')
} finally {
    if ($null -ne $doc) { $doc.Close(0); [Runtime.InteropServices.Marshal]::FinalReleaseComObject($doc) | Out-Null }
    if ($null -ne $word) { $word.Quit(); [Runtime.InteropServices.Marshal]::FinalReleaseComObject($word) | Out-Null }
}
