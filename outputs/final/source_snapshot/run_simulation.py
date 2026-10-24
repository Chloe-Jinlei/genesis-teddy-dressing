"""Image-only teddy dressing reconstruction, using native Genesis PBD dynamics.

No state, asset, parameter or trajectory from the user's original experiment is read.
Only the locally installed upstream Panda URDF is reused.
"""
from __future__ import annotations
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT/'cache/numba'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT/'cache'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'cache/matplotlib'))
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', '1')
import argparse
import json
import hashlib
import shutil
import time
import numpy as np
import genesis as gs
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
from build_assets import build_assets
from native_seam import NativeVelcro
from validate_result import mesh_topology, validate, scalar

DT = .02

def arr(t):
    return t.detach().cpu().numpy() if hasattr(t, 'detach') else np.asarray(t)

def smooth(x):
    x = np.clip(x, 0, 1)
    return x*x*(3-2*x)

def motion(t):
    """Symmetric grasp positions. Explicit open-loop hypothesis, not recovered motion."""
    keys = [
        (0., (.30, -.10, 1.28)),
        (1.5, (.375, -.357, .760)),
        (4.0, (.365, -.23, 1.11)),
        (6.0, (.305, -.09, 1.08)),
        (8.0, (.235, .095, 1.10)),
        (10., (.155, .22, 1.10)),
        (12., (-.018, .194, 1.02)),
        (14., (-.018, .194, 1.02)),
        (16., (.30, .16, 1.28)),
        (18., (.30, .16, 1.28)),
    ]
    for i in range(len(keys)-1):
        if t <= keys[i+1][0]:
            a, p = keys[i]; b, q = keys[i+1]
            v = np.array(p)*(1-smooth((t-a)/(b-a))) + np.array(q)*smooth((t-a)/(b-a))
            left=v*np.array([-1,1,1])
            left[1]-=.012*smooth((t-10)/2)
            return [left, v]
    return [np.array(keys[-1][1])*[-1,1,1], np.array(keys[-1][1])]

def stage(t):
    for end, label in [(1.5,'APPROACH'),(4,'LIFT'),(6,'ALIGN SLEEVE OPENINGS'),
                       (10,'WRAP AROUND TORSO'),(12,'BRING VELCRO TOGETHER'),
                       (14,'UPPER SEAM CONTACT'),(16,'REGRASP LOWER CORNERS'),
                       (19,'CLOSE LOWER SEAM'),(21,'SEAM HOLD'),(23,'FINAL RELEASE'),(100,'FREE SETTLING')]:
        if t < end: return label

def hand_quat(t, side):
    # Turn the fingers toward each other, keeping palms clear of the back seam.
    a=side*np.pi/2*smooth((t-6)/4)
    return [0,float(np.cos(a/2)),0,float(-np.sin(a/2))]

def sleeve_targets(t):
    keys=[(4,(.195,-.23,.99)),(6,(.175,-.06,.940)),
          (8,(.175,.045,.941)),(10,(.155,.058,.985))]
    for (a,p),(b,q) in zip(keys[:-1],keys[1:]):
        if t<=b:
            alpha=smooth((t-a)/(b-a));v=np.array(p)*(1-alpha)+np.array(q)*alpha
            return np.array([v*[-1,1,1],v])
    v=np.array(keys[-1][1]);return np.array([v*[-1,1,1],v])

def build_scene(args, output):
    manifest = build_assets(ROOT/'assets')
    gs.init(backend=gs.cpu if args.backend=='cpu' else gs.gpu, seed=24, logging_level='warning')
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=args.substeps),
        rigid_options=gs.options.RigidOptions(enable_self_collision=False, enable_collision=False),
        pbd_options=gs.options.PBDOptions(particle_size=args.particle_size,
            max_stretch_solver_iterations=args.stretch_iterations, max_bending_solver_iterations=3,
            hash_grid_res=(.48,.48,.48),
            lower_bound=(-1,-.75,.60), upper_bound=(1,.75,1.6)),
        viewer_options=gs.options.ViewerOptions(camera_pos=(1.65,-2.1,1.8),camera_lookat=(0,0,1.0)),
        vis_options=gs.options.VisOptions(show_world_frame=False, ambient_light=(.35,.35,.35)),
        show_viewer=args.viewer)
    def box(pos,size,color,coupled=False):
        return scene.add_entity(gs.morphs.Box(pos=pos,size=size,fixed=True),
            material=gs.materials.Rigid(needs_coup=coupled),surface=gs.surfaces.Rough(color=(*color,1)))
    box((0,0,.725),(1.4,.95,.05),(.52,.35,.21),True)
    for x in [-.59,.59]:
        for y in [-.37,.37]: box((x,y,.35),(.055,.055,.70),(.105,.13,.14))
    scene.add_entity(gs.morphs.Plane(),material=gs.materials.Rigid(needs_coup=False),
                     surface=gs.surfaces.Rough(color=(.17,.19,.21,1)))
    for x,y,c in [(-.47,-.28,(.1,.3,.66)),(.47,-.28,(.72,.12,.08)),
                  (-.29,.30,(.12,.48,.27)),(.29,.30,(.96,.61,.06))]:
        box((x,y,.77),(.04,.04,.04),c)
    # Every bear shape is reconstructed from the supplied silhouette, using ellipsoids.
    for component in manifest['bear']['components']:
        scene.add_entity(gs.morphs.Mesh(file=str(ROOT/'assets/unit_sphere.obj'),
            pos=component['position'], scale=component['scale'],
            euler=component.get('euler_xyz_deg',(0,0,0)),fixed=True,
            convexify=True,decimate=False),
            material=gs.materials.Rigid(needs_coup=component.get('collision',False)),
            surface=gs.surfaces.Rough(color=tuple(component['color'])))
    robots=[]; hands=[]
    for side in [-1,1]:
        box((side*.55,.12,.765),(.16,.17,.03),(.09,.11,.13))
        robot=scene.add_entity(gs.morphs.URDF(file='urdf/panda_bullet/panda.urdf',
            pos=(side*.55,.12,.78),euler=(0,0,0 if side<0 else 180),fixed=True,
            links_to_keep=['panda_hand'],
            collision=False),material=gs.materials.Rigid(needs_coup=False))
        robots.append(robot); hands.append(robot.get_link('panda_hand'))
    cloth=scene.add_entity(gs.morphs.Mesh(file=str(ROOT/'assets/garment_sim.obj'),
        pos=(0,-.28,.758),decimate=False,convexify=False),
        material=gs.materials.PBD.Cloth(rho=.25,stretch_compliance=1e-8,
            bending_compliance=2e-5,stretch_relaxation=args.stretch_relaxation,bending_relaxation=.1,
            static_friction=.35,kinetic_friction=.25,air_resistance=.002),
        surface=gs.surfaces.Rough())
    initial = cloth._particles.copy()
    seam=NativeVelcro(cloth,capture_distance=.018,matching='contact',max_relative_speed=.015,
                     relaxation=args.stretch_relaxation)
    cameras={}
    views={
        'front':((1.55,-2.05,1.65),(0,0,1.02),42),
        'top':((0,-.03,2.75),(0,0,.80),48),
        'back':((0,2.2,1.52),(0,.03,1.01),45),
        'side':((2.15,-.08,1.50),(0,0,.99),43),
        'close':((.95,-1.25,1.45),(0,.06,.98),33),
        'back_close':((.65,1.45,1.25),(0,.07,.97),30),
    }
    for name,(pos,lookat,fov) in views.items():
        cameras[name]=scene.add_camera(res=(960,600),pos=pos,lookat=lookat,fov=fov,GUI=False)
    if args.contact_iterations:
        from pbd_contact import install_interleaved_contact
        contact_metadata=install_interleaved_contact(scene,iterations=args.contact_iterations)
    else:
        contact_metadata={'adapter':None,'schedule':'Unmodified Genesis native PBD/coupling order'}
    (output/'contact_solver.json').write_text(json.dumps(contact_metadata,indent=2))
    print('BUILDING_NATIVE_SCENE',flush=True)
    scene.build()
    seam.init_after_build()
    # Native links are driven by smooth prescribed joint positions. Cloth is NOT pose-driven.
    for robot,hand,p in zip(robots,hands,motion(0)):
        robot.set_dofs_kp(np.array([800,800,800,800,500,500,500,300,300]))
        robot.set_dofs_kv(np.array([80,80,80,80,50,50,50,30,30]))
        q=robot.inverse_kinematics(hand,pos=p,quat=[0,1,0,0],local_point=[0,0,.105],
                                 max_samples=8,max_solver_iters=60)
        q[-2:]=.035
        robot.set_qpos(q); robot.control_dofs_position(q)
    (output/'scene_manifest.json').write_text(json.dumps(manifest,indent=2))
    np.savez_compressed(output/'mesh_rest.npz',particles=initial,faces=cloth._mesh.faces,
                        edges=cloth._edges[:seam.edge_start_local])
    return scene,cloth,seam,robots,hands,cameras,initial,manifest

def frame(camera,t,latched,phase):
    rgb,*_=camera.render()
    im=Image.fromarray(rgb)
    d=ImageDraw.Draw(im)
    d.rectangle((0,0,960,48),fill=(15,22,29))
    d.text((18,10),f'GENESIS  |  IMAGE-ONLY RECONSTRUCTION  |  {phase}',fill=(234,232,221))
    d.text((18,29),f't={t:05.2f}s   native PBD cloth   Velcro bonds: {latched}',fill=(160,186,193))
    return np.asarray(im)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--backend',choices=['cpu','gpu'],default='cpu')
    p.add_argument('--viewer',action='store_true')
    p.add_argument('--duration',type=float,default=25)
    p.add_argument('--render-every',type=int,default=5)
    p.add_argument('--sample-every',type=int,default=1)
    p.add_argument('--detail-video',action='store_true')
    p.add_argument('--output',default='outputs/native_trial')
    p.add_argument('--no-video',action='store_true')
    p.add_argument('--particle-size',type=float,default=.008)
    p.add_argument('--substeps',type=int,default=12)
    p.add_argument('--stretch-iterations',type=int,default=120)
    p.add_argument('--stretch-relaxation',type=float,default=.1)
    p.add_argument('--contact-iterations',type=int,default=8)
    args=p.parse_args(); output=ROOT/args.output; output.mkdir(parents=True,exist_ok=True)
    if min(args.render_every,args.sample_every,args.substeps,args.stretch_iterations)<=0:
        p.error('Sampling periods and solver iteration counts must be positive.')
    source_dir=output/'source_snapshot';source_dir.mkdir(exist_ok=True)
    for source in ROOT.glob('*.py'):shutil.copy2(source,source_dir/source.name)
    (output/'run_config.json').write_text(json.dumps(vars(args),indent=2))
    (output/'source_sha256.json').write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest()
        for f in source_dir.glob('*.py')},indent=2))
    started=time.monotonic()
    scene,cloth,seam,robots,hands,cameras,initial,manifest=build_scene(args,output)
    grips=[];lower_grips=[]
    for side in [-1,1]:
        ids=np.where((initial[:,0]*side>.35)&(initial[:,1]<-.35))[0]
        center=np.array([side*.375,-.357,.758])
        ids=ids[np.argsort(np.linalg.norm(initial[ids]-center,axis=1))[:1]]
        grips.append(ids)
        target=np.array([side*.380,-.203,.758])
        lower_grips.append(np.argsort(np.linalg.norm(initial-target,axis=1))[:1])
    grip_offsets=[initial[g].mean(axis=0)-motion(1.5)[j] for j,g in enumerate(grips)]
    print('SCENE_READY',len(initial),'particles','grips',[g.tolist() for g in grips],flush=True)
    for name,cam in cameras.items():
        rgb,*_=cam.render();Image.fromarray(rgb).save(output/f'initial_{name}.png')
    if not args.no_video:
        writer=imageio.get_writer(output/'simulation.mp4',fps=1/(DT*args.render_every),codec='libx264',quality=8,macro_block_size=2)
        if args.detail_video:
            detail_writer=imageio.get_writer(output/'details.mp4',fps=1/(DT*args.render_every),codec='libx264',quality=8,macro_block_size=2)
    logs=[]; frames=[]; samples=[]; attached=False; released=False; last_stage=''
    top_active=False; lower_active=False; regrasp_started=False; lower_start=None
    released_at=None;release_poses=None
    lower_targets=[np.array([.018,.182,.88]),np.array([-.018,.194,.88])]
    q_samples=[];v_samples=[]
    _,hole_ids,_=mesh_topology(initial,cloth._mesh.faces)
    feedback_offsets=np.zeros((2,3))
    robot_errors=[]
    try:
        for i in range(int(args.duration/DT)+1):
            t=i*DT
            positions=arr(cloth.get_particles_pos())
            velocity=arr(cloth.get_particles_vel())
            seam.update(positions,velocity,step=i,allow_new=(12<=t<14 or t>=19))
            diag=seam.diagnostics();count=diag['active_count']
            if top_active and t>=14 and count>=3:
                for ids in grips:cloth.release_particle(ids)
                top_active=False
                bonded=seam.pairs[seam.active]
                seam_span=np.ptp(positions[bonded].mean(axis=1)[:,2])
                # Regrasp only if the actual contact seam has not already closed
                # the panel height. Pulling an already bonded lower corner is needless.
                material_bins=[np.histogram(initial[bonded[:,j],1],bins=np.linspace(-.37,-.19,9))[0] for j in [0,1]]
                if count>=12 and seam_span>.12 and all(np.count_nonzero(b)>=6 for b in material_bins):
                    released=True;released_at=t;release_poses=motion(14)
                    print('RELEASED_GRASPS_DISTRIBUTED_CONTACT',t,count,flush=True)
                else:
                    regrasp_started=True
                    print('RELEASE_TOP_FOR_REGRASP',t,count,flush=True)
            poses=motion(min(t,14))
            if 4<=t<10:
                holes=np.array([positions[g].mean(axis=0) for g in hole_ids])
                error=sleeve_targets(t)-holes
                feedback_offsets+=np.clip(.05*error,-.002,.002)
                feedback_offsets=np.clip(feedback_offsets,-.16,.16)
            if 4<=t<12:
                weight=1-smooth((t-10)/2)
                poses=[pose+feedback_offsets[j]*weight for j,pose in enumerate(poses)]
            if t<4:
                correction=smooth(t/1.5) if t<1.5 else 1-smooth((t-1.5)/2.5)
                poses=[pose+offset*correction for pose,offset in zip(poses,grip_offsets)]
            if regrasp_started and not lower_active and not released:
                alpha=smooth((t-14)/2)
                poses=[a*(1-alpha)+positions[g].mean(axis=0)*alpha for a,g in zip(motion(14),lower_grips)]
            if lower_active:
                alpha=smooth((t-16)/3)
                poses=[a*(1-alpha)+b*alpha for a,b in zip(lower_start,lower_targets)]
            if released:
                alpha=smooth((t-released_at)/2)
                poses=[a*(1-alpha)+np.array([s*.31,.15,1.26])*alpha for s,a in zip([-1,1],release_poses)]
            for side,robot,hand,pose in zip([-1,1],robots,hands,poses):
                q,error=robot.inverse_kinematics(hand,pos=pose,quat=hand_quat(t,side),local_point=[0,0,.105],
                    init_qpos=robot.get_qpos(),max_samples=4,max_solver_iters=40,return_error=True,pos_tol=.0003,rot_tol=.01)
                if np.linalg.norm(arr(error)[:3])>.002:
                    q,error=robot.inverse_kinematics(hand,pos=pose,quat=None,local_point=[0,0,.105],
                        init_qpos=q,max_samples=4,max_solver_iters=50,return_error=True,pos_tol=.0003)
                q[-2:]=.008 if top_active or lower_active else .035
                robot.set_qpos(q,zero_velocity=True)
                robot.control_dofs_position(q)
                robot_errors.append(float(np.linalg.norm(arr(error)[:3])))
            if not attached and t>=1.5:
                for hand,ids in zip(hands,grips): cloth.fix_particles_to_link(hand.idx,ids)
                attached=True
                top_active=True
                print('NATIVE_GRASP_ATTACHED',t,flush=True)
            if regrasp_started and not lower_active and not released and t>=16:
                lower_start=[positions[g].mean(axis=0) for g in lower_grips]
                for hand,ids in zip(hands,lower_grips):cloth.fix_particles_to_link(hand.idx,ids)
                lower_active=True
                print('NATIVE_LOWER_GRASP',t,flush=True)
            seam_heights=positions[seam.pairs[seam.active]].mean(axis=1)[:,2] if count else np.array([0])
            if lower_active and not released and t>=21 and count>=6 and np.ptp(seam_heights)>.07:
                for ids in lower_grips:cloth.release_particle(ids)
                lower_active=False
                released=True;released_at=t;release_poses=[v.copy() for v in lower_targets]
                print('RELEASED_GRASPS',t,count,flush=True)
            phase=('FREE SETTLING' if t>=released_at+2 else 'FINAL RELEASE') if released else stage(t)
            scene.step()
            if phase!=last_stage:
                print('STAGE',phase,t,diag,flush=True);last_stage=phase
                for name in ['front','close','back']:
                    rgb,*_=cameras[name].render();Image.fromarray(rgb).save(output/f'stage_{i:04d}_{name}.png')
            if i%args.render_every==0:
                if not args.no_video:writer.append_data(frame(cameras['front'],t,count,phase))
                if not args.no_video and args.detail_video:
                    detail_writer.append_data(np.concatenate([frame(cameras[name],t,count,phase) for name in ['close','back_close']],axis=1))
            if i%args.sample_every==0:
                samples.append(arr(cloth.get_particles_pos()).copy());frames.append(t)
                v_samples.append(arr(cloth.get_particles_vel()).copy())
                q_samples.append(np.array([arr(r.get_qpos()) for r in robots]))
                logs.append({'time':t,'stage':phase,'grasp_attached':top_active or lower_active,
                             'velcro':diag,'ik_error_m':max(robot_errors[-2:]),
                             'hole_centers':[positions[g].mean(axis=0).tolist() for g in hole_ids],
                             'feedback_offsets':feedback_offsets.tolist()})
            if i%100==0:print('PROGRESS',i,'/',int(args.duration/DT),'wall',round(time.monotonic()-started,1),flush=True)
        for name,cam in cameras.items():
            rgb,*_=cam.render();Image.fromarray(rgb).save(output/f'final_{name}.png')
    finally:
        if not args.no_video:writer.close()
        if not args.no_video and args.detail_video:detail_writer.close()
    np.savez_compressed(output/'trajectory.npz',time=np.array(frames),particles=np.array(samples),
                        faces=cloth._mesh.faces,rest=initial,robot_qpos=np.array(q_samples),
                        velocities=np.array(v_samples),state_time_offset=DT)
    np.savez_compressed(output/'seam_state.npz',pairs=seam.pairs,active=seam.active,
                        rest_gap=seam.rest_gap)
    (output/'seam_events.json').write_text(json.dumps(seam.events,indent=2))
    (output/'metrics.json').write_text(json.dumps({'genesis_version':gs.__version__,
        'backend':args.backend,'particle_count':len(initial),'released':released,
        'actual_genesis_backend':str(gs.backend),'torch_device':str(gs.device),
        'solver_dt_s':DT,'state_time_note':'Time labels are control-step starts; saved particle states are after the following dt.',
        'solver_parameters':vars(args),
        'max_ik_position_error_m':max(robot_errors),'velcro':seam.diagnostics(),
        'wall_seconds':time.monotonic()-started,'logs':logs},indent=2))
    report=validate(output)
    (output/'validation.json').write_text(json.dumps(report,indent=2,default=scalar))
    print('VALIDATION',report.get('status'),report.get('failed_checks'),flush=True)
    print('FINISHED',output,flush=True)

if __name__=='__main__':main()
