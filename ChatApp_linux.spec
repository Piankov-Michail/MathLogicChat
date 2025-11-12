# ChatApp_linux.spec

import os
from PyInstaller.utils.hooks import collect_data_files

datas = collect_data_files('kivy')

fonts_dir = os.path.join(os.getcwd(), 'fonts')
if os.path.exists(fonts_dir):
    datas.append((fonts_dir, 'fonts'))
else:
    print("Warning: 'fonts' directory not found!")

a = Analysis(
    ['src/main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'kivy',
        'kivy.uix.boxlayout',
        'kivy.uix.textinput',
        'kivy.uix.button',
        'kivy.uix.label',
        'kivy.uix.scrollview',
        'kivy.uix.popup',
        'kivy.uix.gridlayout',
        'kivy.clock',
        'kivy.metrics',
        'kivy.core.window',
        'kivy.core.clipboard',
        'kivy.core.text',
        'kivy.graphics',
        'openai',
        'typing',
        'threading',
        'os',
        'sys',
        'time',
        'json',
        'uuid',
        'pathlib',
        'resolution_method'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='ChatApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
)