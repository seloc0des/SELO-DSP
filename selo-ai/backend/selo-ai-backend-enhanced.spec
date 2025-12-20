# -*- mode: python ; coding: utf-8 -*-
# SELO AI Backend PyInstaller Spec (Enhanced Security for Beta)

# Note: PyInstaller v6.0+ removed bytecode encryption (cipher argument)
# We rely on bytecode optimization and binary stripping for obfuscation

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('constraints', 'constraints'), ('prompt', 'prompt'), ('core', 'core')],
    hiddenimports=[
        'uvicorn', 'fastapi', 'socketio', 'sqlalchemy', 
        'hashlib', 'uuid', 'platform', 'cryptography', 'secrets'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=2,  # Maximum bytecode optimization (harder to decompile)
)

pyz = PYZ(
    a.pure,
    a.zipped_data
)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='selo-ai-backend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,  # Strip symbols from binary
    upx=True,  # Compress with UPX
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
