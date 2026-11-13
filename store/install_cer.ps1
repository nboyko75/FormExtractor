## with admin privileges

Import-Certificate -FilePath "D:\Projects\FormExtractor\store\FormExtractor.Dev.cer" -CertStoreLocation Cert:\LocalMachine\TrustedPeople

Add-AppxPackage -Path "D:\Projects\FormExtractor\store\FormExtractor.Dev.msix"