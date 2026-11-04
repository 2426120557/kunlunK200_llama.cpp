"""Bounded tests with existing thermal guard and device-state fail-stop."""
from pathlib import Path
import hashlib,json,os,signal,subprocess,threading,time,re
from bench_affinity import prepare
ROOT=OUT=CHECK_USERS=None
SOURCE_ROOTS=[]

def configure(workspace,results,source_roots=(),check_users=None):
    global ROOT,OUT,CHECK_USERS,SOURCE_ROOTS
    ROOT=Path(workspace).resolve();OUT=Path(results).resolve()
    CHECK_USERS=Path(check_users).resolve() if check_users else OUT/'check_users.py'
    SOURCE_ROOTS=[Path(p).resolve() for p in source_roots]
    if not ROOT.is_dir() or not CHECK_USERS.is_file() or not (ROOT/'install/bin/k200-thermal-guard').is_file():
        raise ValueError('Existing workspace, user checker and thermal guard are required')
    OUT.mkdir(parents=True,exist_ok=True)

def states():
    result={}
    for pd in [0,1]:
        try:result[pd]=Path(f'/proc/xpu/dev{pd}/state').read_text().strip()
        except OSError as error:result[pd]='unavailable: '+str(error)
    return result

def device_healthy():
    try:
        return all(v=='RUNNING' for v in states().values()) and all(Path(f'/proc/xpu/dev{pd}/errinfo').read_text().strip()=='no error' and not Path(f'/proc/xpu/dev{pd}/errtask').read_text().strip() for pd in [0,1])
    except OSError:
        return False

def run(name,command,env=None,seconds=180):
    if ROOT is None or OUT is None:raise ValueError("Call configure(workspace, results) first")
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,119}',name):raise ValueError('Invalid run name')
    env=dict(env or {})
    logs=OUT/'logs';logs.mkdir(exist_ok=True)
    if any(p.exists() for p in [OUT/'XPU_ABORT',OUT/'logs/THERMAL_ABORT',ROOT/'results/THERMAL_ABORT']):raise RuntimeError('Persistent abort latch')
    try:
        preflight={pd:{field:Path(f'/proc/xpu/dev{pd}/{field}').read_text().strip() for field in ['state','errinfo','errtask']} for pd in [0,1]}
    except OSError as error:
        (OUT/'XPU_ABORT').write_text(json.dumps({'phase':'preflight_read_error','name':name,'error':str(error)}))
        raise RuntimeError('Device preflight unavailable') from error
    if any(v['state']!='RUNNING' or v['errinfo']!='no error' or v['errtask'] for v in preflight.values()):
        (OUT/'XPU_ABORT').write_text(json.dumps({'phase':'preflight','name':name,'states':preflight},indent=2))
        raise RuntimeError('Device state/error preflight failed')
    subprocess.run(['sudo','-n','python3',str(CHECK_USERS)],check=True)
    prior=list(logs.glob(name+'*'))
    if prior:
        raise FileExistsError('Existing run logs are protected: '+name)
    policy=env.pop('LLAMA_BENCH_MAIN_CPU_POLICY','off')
    env,affinity_plan=prepare(policy,env)
    (logs/(name+'-affinity.json')).write_text(json.dumps(affinity_plan,indent=2))
    args=list(map(str,command));done=threading.Event()
    def watch():
        while not done.wait(.1):
            if device_healthy():continue
            (OUT/'XPU_ABORT').write_text('device state error during '+name+'\n')
            # Kill only this test's exact argv, after verifying ancestry.
            for proc in Path('/proc').iterdir():
                if not proc.name.isdecimal():continue
                try:
                    argv=[x.decode() for x in proc.joinpath('cmdline').read_bytes().split(b'\0') if x]
                    if argv!=args:continue
                    pid=int(proc.name);ancestor=pid
                    for _ in range(8):
                        stat=Path(f'/proc/{ancestor}/stat').read_text();ancestor=int(stat[stat.rfind(')')+2:].split()[1])
                        if ancestor==os.getpid():os.kill(pid,signal.SIGKILL);break
                        if ancestor<=1:break
                except (OSError,ValueError):continue
            return
    e=os.environ.copy();e.update(env or {});e['KUNLUN_LOG_DIR']=str(logs);e['K200_WORKSPACE']=str(ROOT)
    begin=time.monotonic()
    def host_watch():
        with (logs/(name+'-host.jsonl')).open('w') as f:
            while not done.is_set():
                row={'elapsed':time.monotonic()-begin,'utc_ns':time.time_ns(),'loadavg':Path('/proc/loadavg').read_text().strip(),'cpu_stat':Path('/proc/stat').read_text().splitlines()[0]}
                row['mhz']=[float(v.split(':',1)[1]) for v in Path('/proc/cpuinfo').read_text().splitlines() if v.startswith('cpu MHz')]
                try:
                    queue=[os.getpid()];visited=set()
                    for _ in range(64):
                        if not queue:break
                        pid=queue.pop(0)
                        if pid in visited:continue
                        visited.add(pid)
                        cmd=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
                        if cmd and cmd[0].decode(errors='replace')==args[0]:
                            st=Path(f'/proc/{pid}/stat').read_text();fields=st[st.rfind(')')+2:].split();cpu=int(fields[36])
                            row['process']={'pid':pid,'cpu':cpu,'utime':int(fields[11]),'stime':int(fields[12]),'mhz':row['mhz'][cpu] if cpu<len(row['mhz']) else None,'schedstat':Path(f'/proc/{pid}/schedstat').read_text().strip()};break
                        children=Path(f'/proc/{pid}/task/{pid}/children')
                        if children.exists():queue.extend(int(x) for x in children.read_text().split())
                except (OSError,ValueError,IndexError):pass
                row['cpu_detail']=[v for v in Path('/proc/stat').read_text().splitlines() if re.match(r'cpu\d+ ',v)]
                row['xpu_irq_effective']={}
                for irq in affinity_plan['xpu_irqs']:
                    path=Path(f"/proc/irq/{irq['irq']}/effective_affinity_list")
                    if path.exists():row['xpu_irq_effective'][str(irq['irq'])]=path.read_text().strip()
                pressure=Path('/proc/pressure/cpu')
                if pressure.exists():row['cpu_pressure']=pressure.read_text().strip()
                f.write(json.dumps(row)+'\n');f.flush()
                if done.wait(1):break
    monitor=threading.Thread(target=watch,daemon=True);monitor.start()
    host=threading.Thread(target=host_watch,daemon=True);host.start()
    print('RUN',name,flush=True)
    try:
        with (logs/(name+'.txt')).open('w') as f:
            rc=subprocess.run([str(ROOT/'install/bin/k200-thermal-guard'),'--name',name,'--seconds',str(seconds),'--',*args],env=e,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT).returncode
    finally:done.set();monitor.join(timeout=1);host.join(timeout=2)
    final=states();failed=rc!=0 or any(v!='RUNNING' for v in final.values())
    meta={'name':name,'command':args,'environment':env,'affinity':affinity_plan,'returncode':rc,'elapsed_s':time.monotonic()-begin,'states':final,
          'binary_sha256':{str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in args[:1] if Path(p).is_file()}}
    (logs/(name+'.json')).write_text(json.dumps(meta,indent=2))
    text=(logs/(name+'.txt')).read_text(errors='replace')
    if failed:
        (OUT/'XPU_ABORT').write_text(json.dumps(meta,indent=2))
        with (logs/(name+'-dmesg.txt')).open('w') as f:subprocess.run(['sudo','-n','dmesg','-T'],stdout=f,stderr=subprocess.STDOUT)
        import shutil
        incident=OUT/'incidents'/name;incident.mkdir(parents=True,exist_ok=True)
        for pd in [0,1]:
            for item in ['state','errinfo','errtask','sessions']:
                p=Path(f'/proc/xpu/dev{pd}')/item
                if p.exists():(incident/f'pd{pd}-{item}.txt').write_text(p.read_text(errors="replace"))
        for root in SOURCE_ROOTS:
            with (incident/(root.name+'-diff.patch')).open('w') as f:subprocess.run(['git','diff','HEAD','--binary'],cwd=root,stdout=f)
        for i,p in enumerate(args[:1]):
            if Path(p).is_file():shutil.copy2(p,incident/f'command-{i}.binary')
        kernel_dir=Path(e.get('KUNLUN_KERNEL_DIR',OUT/'install/lib/kunlunk200/kernels'))
        image_dir=incident/'kernels';image_dir.mkdir(exist_ok=True)
        for src in kernel_dir.glob('kunlun-*'):
            if src.is_file() and src.suffix in ['.bin','.entry']:shutil.copy2(src,image_dir/src.name)
        library=kernel_dir.parent.parent/'libkunlunk200.so'
        if library.exists():shutil.copy2(library,incident/'libkunlunk200.so')
        print(text[-3000:],flush=True);raise RuntimeError(f'{name} failed; later tests stopped')
    return text
