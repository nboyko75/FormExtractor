# MSIX packaging

This follows DocExplorer's workflow: build the app with `build.cmd`, then package
the existing `dist/FormExtractor/` folder. Packaging does not rebuild the executable.
The scripts require Windows PowerShell 5.1 and the Windows 10/11 SDK containing
`makeappx.exe`, `makepri.exe`, and (for development signing) `signtool.exe`.
The newest complete installed SDK is selected automatically.

## Development installer

```powershell
.\build.cmd
.\build_dev.cmd
# Optional version override:
.\build_dev.cmd -Version 1.0.0.1
```

Outputs:

- `store/FormExtractor.Dev.msix`: signed development installer.
- `store/FormExtractor.Dev.cer`: public signing certificate.

As in DocExplorer, the signing certificate subject matches the manifest Publisher.
The script creates or reuses a certificate named `FormExtractor MSIX Development`
in `Cert:\CurrentUser\My`. Its private key is non-exportable and stays outside the
repository. Certificates last one year and are reused while more than 30 days
remain. No certificate password or PFX file is needed.

To trust the certificate for local testing, run in **elevated PowerShell**:

```powershell
Import-Certificate -FilePath .\store\FormExtractor.Dev.cer -CertStoreLocation Cert:\LocalMachine\TrustedPeople
```

Then install as your normal user:

```powershell
Add-AppxPackage -Path .\store\FormExtractor.Dev.msix
```

The build prints absolute versions of these commands. It does not change trust or
install the app automatically. If a renewed certificate is created, trust its new
public `.cer` before installing the newly signed package.

## Partner Center installer

`store/AppxManifest.xml` uses the confirmed Partner Center package identity
`NickBoiko.FormExtractor`. The relevant **Product identity** fields are:

- `Identity/@Name`: the exact Package/Identity/Name, not the Store product ID.
- `Identity/@Publisher`: the exact Package/Identity/Publisher.
- `Properties/PublisherDisplayName` and both display names: match your listing.

Publisher and publisher display name are copied from DocExplorer's
account; verify them if using a different account. The MSIX version starts at
`1.0.0.0`, separately from the Python package version. Increase it for subsequent
submissions; Store builds require a nonzero first component and a zero fourth component.

```powershell
.\build_store.cmd -Version 1.0.0.0
# Alternatively override identity for this build without changing the manifest:
.\build_store.cmd -IdentityName YOUR.EXACT.PACKAGE.NAME -Version 1.0.0.0
```

Output: `store/FormExtractor.msix`. Upload this file to **Packages** in your
Partner Center submission. Complete the listing and certification requirements,
including the reason for the desktop application's `runFullTrust` capability.
The script prepares the package and does not upload or publish it automatically,
matching DocExplorer's approach. No `.msixupload` wrapper is required for this flow.

The Store artifact is unsigned: Microsoft signs it during submission, so it does
not use the local developer certificate. See Microsoft's
[MSIX signing guide](https://learn.microsoft.com/windows/msix/package/sign-msix-package-guide)
and [package requirements](https://learn.microsoft.com/en-us/windows/apps/publish/publish-your-app/msix/app-package-requirements).

## Assets and build behavior

`Assets/` contains FormExtractor icons at the manifest sizes and the same unplated
target sizes used by DocExplorer. To regenerate them:

```powershell
.\.venv\Scripts\python.exe store\generate_assets.py
```

Both modes generate `resources.pri` with DocExplorer's resource index configuration
so Windows can select the target-size icon variants. The package architecture is
`x64`, matching `build.cmd`. Version and identity overrides affect only the staged
manifest. Each build uses a unique staging directory that is removed afterwards;
an existing MSIX is replaced only after packaging and any signing succeed.

The package includes the full PyInstaller folder, including Paddle's libraries.
OCR models are downloaded on first recognition unless already cached. Test document
preview and recognition in an installed development package before Store submission.
