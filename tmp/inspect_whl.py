import zipfile
from pathlib import Path
with zipfile.ZipFile('tmp/mediapipe-0.10.32-py3-none-win_amd64.whl') as z:
    for name in z.namelist():
        if name.endswith('METADATA'):
            print(name)
            for line in z.read(name).decode().splitlines():
                if line.startswith('Requires-Dist') or line.startswith('Requires-Python'):
                    print(line)
            break
