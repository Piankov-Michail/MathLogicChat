# ChatApp_windows.spec

import os
from kivy_deps import sdl2, glew
from PyInstaller.utils.hooks import collect_data_files

datas = collect_data_files('kivy')

fonts_dir = os.path.join(os.getcwd(), 'fonts')
if os.path.exists(fonts_dir):
    datas.append((fonts_dir, 'fonts'))
else:
    print("'fonts'not found!.")

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
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
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
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    *[Tree(p) for p in (sdl2.dep_bins + glew.dep_bins)],
    strip=False,
    upx=True,
    name='ChatApp'
)