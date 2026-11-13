#Requires -Version 5.1
<#
.SYNOPSIS
Packages the existing PyInstaller output with a development certificate signature.
.EXAMPLE
.\build_dev.cmd -Version 1.0.0.1
#>
[CmdletBinding()]
param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$Version,
    [string]$IdentityName,
    [string]$Publisher
)

$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'build_msix.ps1') @PSBoundParameters -Development
