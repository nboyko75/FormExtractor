#Requires -Version 5.1
<#
.SYNOPSIS
Packages existing PyInstaller output for Partner Center, or signs a development MSIX.
.EXAMPLE
.\build_store.cmd -IdentityName YOUR.PARTNER.CENTER.NAME -Version 1.0.0.0
.NOTES
Run build.cmd first. Store packages are unsigned; Microsoft signs them on submission.
This script creates an upload artifact; it does not submit or publish the app.
#>
[CmdletBinding()]
param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$Version,
    [string]$IdentityName,
    [string]$Publisher,
    [switch]$Development
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$root = (Resolve-Path -LiteralPath $ProjectRoot).Path
$storeDir = Join-Path $root 'store'
$manifestPath = Join-Path $storeDir 'AppxManifest.xml'
[xml]$manifest = Get-Content -LiteralPath $manifestPath -Raw
$ns = New-Object System.Xml.XmlNamespaceManager($manifest.NameTable)
$ns.AddNamespace('m', $manifest.DocumentElement.NamespaceURI)
$identity = $manifest.SelectSingleNode('/m:Package/m:Identity', $ns)
if (-not $identity) { throw 'AppxManifest.xml must contain a Package/Identity.' }
if ($Version) { $identity.SetAttribute('Version', $Version) }
if ($IdentityName) { $identity.SetAttribute('Name', $IdentityName) }
if ($Publisher) { $identity.SetAttribute('Publisher', $Publisher) }
$packageVersion = $identity.GetAttribute('Version')
if ($packageVersion -notmatch '^\d+\.\d+\.\d+\.\d+$' -or
    @($packageVersion.Split('.') | Where-Object { [double]$_ -gt 65535 }).Count -gt 0) {
    throw 'Version must have four numeric components between 0 and 65535.'
}
if (-not $Development) {
    if ($identity.Name -eq 'FormExtractor.Development') {
        throw 'Set Identity Name in store/AppxManifest.xml to FormExtractor''s exact Partner Center Product identity, or pass -IdentityName. The development placeholder cannot be submitted.'
    }
    if ([int]$packageVersion.Split('.')[3] -ne 0) {
        throw 'Partner Center requires the fourth version component to be zero.'
    }
    if ([int]$packageVersion.Split('.')[0] -eq 0) {
        throw 'Partner Center requires a nonzero first version component.'
    }
}
if ([string]::IsNullOrWhiteSpace($identity.Publisher)) {
    throw 'The manifest Identity must have a Publisher matching the signing account.'
}
if ($identity.ProcessorArchitecture -ne 'x64') {
    throw 'This packaging workflow requires ProcessorArchitecture="x64".'
}
$appDir = Join-Path $root 'dist\FormExtractor'
if (-not (Test-Path -LiteralPath (Join-Path $appDir 'FormExtractor.exe') -PathType Leaf)) {
    throw 'PyInstaller output is missing at dist\FormExtractor\FormExtractor.exe. Run build.cmd first.'
}
$assetsDir = Join-Path $storeDir 'Assets'
$requiredAssets = @('StoreLogo.png', 'Logo44.png', 'Logo150.png') + @(
    16, 32, 48, 256 | ForEach-Object { "Logo44.targetsize-$($_)_altform-unplated.png" }
)
foreach ($asset in $requiredAssets) {
    if (-not (Test-Path -LiteralPath (Join-Path $assetsDir $asset) -PathType Leaf)) {
        throw "Required MSIX icon is missing: $asset. Run .venv\Scripts\python.exe store\generate_assets.py."
    }
}
$priConfig = Join-Path $storeDir 'priconfig.xml'
if (-not (Test-Path -LiteralPath $priConfig -PathType Leaf)) {
    throw "Resource index configuration is missing: $priConfig"
}
$sdkBin = Join-Path ${env:ProgramFiles(x86)} 'Windows Kits\10\bin'
$sdk = Get-ChildItem -LiteralPath $sdkBin -Directory -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Name -match '^\d+\.\d+\.\d+\.\d+$' -and
        (Test-Path -LiteralPath (Join-Path $_.FullName 'x64\makeappx.exe')) -and
        (Test-Path -LiteralPath (Join-Path $_.FullName 'x64\makepri.exe')) -and
        ((-not $Development) -or (Test-Path -LiteralPath (Join-Path $_.FullName 'x64\signtool.exe')))
    } | Sort-Object { [version]$_.Name } -Descending | Select-Object -First 1
if (-not $sdk) { throw 'Install the Windows 10/11 SDK with makeappx.exe, makepri.exe and signtool.exe.' }
$makeAppx = Join-Path $sdk.FullName 'x64\makeappx.exe'
$makePri = Join-Path $sdk.FullName 'x64\makepri.exe'
$signTool = Join-Path $sdk.FullName 'x64\signtool.exe'
$packageName = if ($Development) { 'FormExtractor.Dev' } else { 'FormExtractor' }
$outputPath = Join-Path $storeDir "$packageName.msix"
$certificatePath = Join-Path $storeDir 'FormExtractor.Dev.cer'

# Stage each invocation separately and only publish its output after all tools succeed.
$layout = Join-Path $storeDir ('PackageLayout-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $layout | Out-Null
try {
    $payload = Join-Path $layout 'Payload'
    New-Item -ItemType Directory -Path $payload | Out-Null
    Get-ChildItem -LiteralPath $appDir -Force | Copy-Item -Destination $payload -Recurse -Force
    Copy-Item -LiteralPath $assetsDir -Destination (Join-Path $payload 'Assets') -Recurse -Force
    $stagedManifest = Join-Path $payload 'AppxManifest.xml'
    $manifest.Save($stagedManifest)
    & $makePri new /pr $payload /cf $priConfig /mn $stagedManifest /of (Join-Path $payload 'resources.pri') /o
    if ($LASTEXITCODE -ne 0) { throw 'makepri failed while indexing MSIX icons.' }
    $stagedPackage = Join-Path $layout "$packageName.msix"
    & $makeAppx pack /d $payload /p $stagedPackage /o
    if ($LASTEXITCODE -ne 0) { throw 'makeappx failed to create the MSIX package.' }

    if ($Development) {
        # Match DocExplorer: non-exportable private key in the current user's store.
        $friendlyName = 'FormExtractor MSIX Development'
        $certificate = Get-ChildItem Cert:\CurrentUser\My |
            Where-Object {
                $_.FriendlyName -eq $friendlyName -and
                $_.Subject -ceq $identity.Publisher -and $_.HasPrivateKey -and
                $_.NotBefore -le (Get-Date) -and $_.NotAfter -gt (Get-Date).AddDays(30) -and
                @($_.EnhancedKeyUsageList | Where-Object { $_.ObjectId -eq '1.3.6.1.5.5.7.3.3' }).Count -gt 0
            } | Sort-Object NotAfter -Descending | Select-Object -First 1
        if (-not $certificate) {
            $certificate = New-SelfSignedCertificate -Type Custom `
                -Subject $identity.Publisher -FriendlyName $friendlyName `
                -CertStoreLocation 'Cert:\CurrentUser\My' `
                -KeyAlgorithm RSA -KeyLength 2048 -HashAlgorithm SHA256 `
                -KeyUsage DigitalSignature -KeyExportPolicy NonExportable `
                -NotAfter (Get-Date).AddYears(1) `
                -TextExtension @('2.5.29.37={text}1.3.6.1.5.5.7.3.3', '2.5.29.19={text}')
        }
        & $signTool sign /fd SHA256 /s My /sha1 $certificate.Thumbprint $stagedPackage
        if ($LASTEXITCODE -ne 0) { throw 'signtool failed to sign the development MSIX.' }
        Export-Certificate -Cert $certificate -FilePath $certificatePath -Force | Out-Null
    }
    Move-Item -LiteralPath $stagedPackage -Destination $outputPath -Force
}
finally {
    $resolvedLayout = [IO.Path]::GetFullPath($layout)
    if ([IO.Path]::GetDirectoryName($resolvedLayout) -ne [IO.Path]::GetFullPath($storeDir) -or
        [IO.Path]::GetFileName($resolvedLayout) -notmatch '^PackageLayout-[0-9a-f]{32}$') {
        throw "Refusing to remove unexpected staging directory: $resolvedLayout"
    }
    Remove-Item -LiteralPath $resolvedLayout -Recurse -Force
}

if ($Development) {
    Write-Host "Signed development MSIX: $outputPath"
    Write-Host "Public certificate: $certificatePath (expires $($certificate.NotAfter))"
    Write-Host 'To trust the developer certificate, run in an elevated PowerShell window:'
    Write-Host "Import-Certificate -FilePath '$($certificatePath.Replace("'", "''"))' -CertStoreLocation Cert:\LocalMachine\TrustedPeople"
    Write-Host 'Then install as your normal user:'
    Write-Host "Add-AppxPackage -Path '$($outputPath.Replace("'", "''"))'"
} else {
    Write-Host "Partner Center package: $outputPath"
    Write-Host "Identity: $($identity.Name); Publisher: $($identity.Publisher); Version: $packageVersion"
    Write-Host 'Upload this MSIX to the Packages page of your Partner Center submission.'
    Write-Host 'Microsoft signs Store packages. This script does not upload or publish automatically.'
}
