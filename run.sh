export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$(python - <<'PY'
import site
from pathlib import Path
p = Path(site.getsitepackages()[0])
print(':'.join(str(x) for x in sorted(p.glob('nvidia/*/lib'))))
PY
)"
#python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
#python main.py finger.jpeg
python main.py match sahibind4.jpeg sahibind5.jpeg