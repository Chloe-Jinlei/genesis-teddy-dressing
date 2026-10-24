import os
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/codex-genesis-numba')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/codex-genesis-cache')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/codex-genesis-mpl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', '1')
import genesis as gs
from PIL import Image
gs.init(backend=gs.cpu, logging_level='warning', seed=0)
s=gs.Scene(show_viewer=False)
s.add_entity(gs.morphs.Plane())
s.add_entity(gs.morphs.Sphere(radius=.15,pos=(0,0,.15)),surface=gs.surfaces.Default(color=(.6,.3,.1,1)))
c=s.add_camera(res=(640,360),pos=(1,-1,.7),lookat=(0,0,.1),GUI=False)
s.build()
for _ in range(4): s.step()
rgb,*_=c.render()
Image.fromarray(rgb).save('outputs/smoke.png')
print('GENESIS_CPU_RENDER_OK',flush=True)
