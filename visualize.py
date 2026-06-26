"""
visualize.py
============

Trains the Selective Maturation Race (8 lines of a 3x3 grid: 3 rows, 3 cols,
2 diagonals) in both substrate modes, records the step-by-step network
dynamics, and emits a single self-contained interactive HTML page at
``snn_visualization.html`` -- vanilla JS + <canvas>, no external assets.

The page renders the *actual* network:
  * the 9 input pixels,
  * all 64 L2 excitatory neurons (membrane level + spikes, animated),
  * the learned feedforward weight connections,
  * the global inhibitory interneuron (the dense lateral -5.0 squeeze) and its
    fan-out, pulsing on every spike,
  * the per-neuron self-excitation (+2.0) loops,
plus a play/scrub timeline so you can watch each primitive get claimed, and the
post-hoc analysis panels (receptive fields, cross-response matrix, race order).

    python visualize.py
"""

from __future__ import annotations

import json

import numpy as np

from snn_layer import SelfOrganizingSNNLayer, SNNConfig
from scale_run_8line import build_line_patterns


# --------------------------------------------------------------------------- #
#  Train one network; optionally record per-step dynamics for every block      #
# --------------------------------------------------------------------------- #
def train_capture(sparse: bool, seed: int = 0, record: bool = False):
    pats, names = build_line_patterns()
    cfg = SNNConfig(input_dim=pats.shape[1], n_neurons=64,
                    seed=seed, sparse_substrate=sparse)
    L = SelfOrganizingSNNLayer(cfg)

    claimed: dict[int, int] = {}
    timeline, traces = [], []

    for epoch in range(60):
        for pid in range(len(pats)):
            if pid in claimed:
                continue
            matured_before = sorted(claimed.values())
            L.reset_transient()
            winner, latency, steps = None, cfg.max_block_steps, []
            for t in range(cfg.max_block_steps):
                L.step(pats[pid])
                if record:
                    plastic = (L.mature_mask == 0)
                    sp = np.where(L.spikes > 0)[0]
                    vq = np.clip(np.round(100.0 * L.v / cfg.theta_rest),
                                 0, 160).astype(int)
                    steps.append(dict(
                        s=[int(x) for x in sp],
                        vq=[int(x) for x in vq],
                        nspk=int((L.spikes * plastic).sum()),
                        numax=round(float((L.nu * plastic).max()), 3),
                    ))
                winner = L.try_mature(pid)
                if winner is not None:
                    latency = t + 1
                    if cfg.sparse_substrate:
                        L.reset_losers(winner)
                    break
            if winner is not None:
                claimed[pid] = int(winner)
                timeline.append(dict(pid=pid, name=names[pid], epoch=epoch,
                                     winner=int(winner), latency=latency,
                                     pool=int(L.mature_mask.sum())))
                if record:
                    traces.append(dict(pid=pid, name=names[pid],
                                       winner=int(winner), mature_step=latency,
                                       matured_before=matured_before,
                                       steps=steps))
        if len(claimed) == len(pats):
            break

    # ---- final state harvest ------------------------------------------- #
    neurons = []
    for i in range(cfg.n_neurons):
        col = L.W[:, i]
        mass = float(col.sum())
        sel = float(col.max() / (mass / cfg.input_dim))
        neurons.append(dict(
            i=i, mature=int(L.mature_mask[i]), pattern=int(L.tuned_pattern[i]),
            mass=round(mass, 3), sel=round(sel, 3),
            rf=[round(float(v), 4) for v in col],
        ))

    resp = np.stack([L.response(p) for p in pats])
    owners = [claimed[p] for p in range(len(pats))]
    xresp = resp[:, owners]
    diag = np.diag(xresp)
    off = xresp - np.diag(diag)

    out = dict(
        sparse=sparse, neurons=neurons, timeline=timeline, owners=owners,
        xresp=[[round(float(v), 3) for v in r] for r in xresp],
        on_min=round(float(diag.min()), 3),
        off_max=round(float(off.max()), 3),
        margin=round(float(diag.min() - off.max()), 3),
        tuned_plastic=int(sum(1 for n in neurons
                              if not n["mature"] and n["sel"] > 2.5)),
        max_mass=round(float(L.W.sum(0).max()), 2),
        W=[[round(float(v), 4) for v in row] for row in L.W],   # 9 x 64
    )
    if record:
        out["traces"] = traces
    return out


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Selective Maturation Race &mdash; live SNN</title>
<style>
  :root{--bg:#0d1117;--panel:#161b22;--line:#30363d;--txt:#e6edf3;
        --muted:#8b949e;--accent:#58a6ff;--good:#3fb950;--inh:#f85149}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--txt);
       font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
  header{padding:20px 28px;border-bottom:1px solid var(--line)}
  h1{margin:0 0 4px;font-size:21px}
  h2{font-size:14px;letter-spacing:.04em;text-transform:uppercase;
     color:var(--muted);margin:0 0 14px}
  .sub{color:var(--muted);font-size:13px}
  main{max-width:1200px;margin:0 auto;padding:22px 28px 60px}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;
         padding:18px 20px;margin-bottom:20px}
  .row{display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start}
  .stat{background:#0d1117;border:1px solid var(--line);border-radius:8px;
        padding:9px 13px;min-width:110px}
  .stat b{display:block;font-size:20px}.stat span{color:var(--muted);font-size:12px}
  button{background:#21262d;color:var(--txt);border:1px solid var(--line);
         border-radius:7px;padding:7px 13px;cursor:pointer;font-size:13px}
  button:hover{border-color:var(--accent)}
  button.on{background:var(--accent);color:#0d1117;font-weight:600;border-color:var(--accent)}
  select,input[type=range]{accent-color:var(--accent)}
  select{background:#21262d;color:var(--txt);border:1px solid var(--line);
         border-radius:7px;padding:6px 10px}
  canvas{width:100%;height:auto;display:block;border-radius:8px;background:#0a0d12}
  .ctl{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
  .readout{display:flex;gap:18px;flex-wrap:wrap;margin-top:12px;font-size:13px}
  .readout b{font-variant-numeric:tabular-nums}
  .legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin-top:10px}
  .legend i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:5px;vertical-align:-1px}
  .grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:2px;width:48px}
  .grid3 i{aspect-ratio:1;border-radius:1px;background:#0d1117}
  .rfwrap{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}
  .rfcard{display:flex;flex-direction:column;align-items:center;gap:6px;background:#0d1117;
          border:1px solid var(--line);border-radius:8px;padding:10px}
  .rfcard .grid3{width:60px}
  table.x{border-collapse:collapse;font-size:11px}
  table.x td,table.x th{border:1px solid var(--line);width:48px;height:32px;text-align:center;
        color:#0d1117;font-weight:600}
  table.x th{background:transparent;color:var(--muted)}
  .toggle{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden}
  .toggle button{border:0;border-radius:0}
  .tl{display:flex;flex-direction:column;gap:6px}
  .tl .bar{display:flex;align-items:center;gap:10px}
  .tl .name{width:54px;font-size:12px}
  .tl .track{flex:1;background:#0d1117;border-radius:4px;height:15px}
  .tl .fill{height:100%;border-radius:4px}.tl .lat{font-size:11px;color:var(--muted);width:74px}
  code{background:#0d1117;padding:1px 5px;border-radius:4px;color:var(--accent)}
</style>
</head>
<body>
<header>
  <h1>Selective Maturation Race &mdash; live spiking network</h1>
  <div class="sub">8 lines of a 3&times;3 grid (3 rows + 3 cols + 2 diagonals) &middot;
     L2 pool N=64 &middot; watch each primitive get claimed and retired.</div>
</header>
<main>
  <div class="panel">
    <div class="row" id="stats"></div>
  </div>

  <div class="panel">
    <h2>Live network &mdash; inputs &middot; excitatory pool &middot; global inhibition</h2>
    <div class="ctl">
      <button id="play">&#9654; Play</button>
      <button id="stepb">Step &raquo;</button>
      <button id="reset">&#8634; Restart block</button>
      <label class="sub">block&nbsp;
        <select id="block"></select></label>
      <label class="sub">speed&nbsp;
        <input type="range" id="speed" min="1" max="30" value="10"></label>
      <button id="alledges">Show all weights</button>
      <input type="range" id="scrub" min="0" max="1" value="0" style="flex:1;min-width:160px">
    </div>
    <canvas id="net" width="1140" height="600"></canvas>
    <div class="readout">
      <span>step <b id="rStep">0</b>/<b id="rTot">0</b></span>
      <span>presented&nbsp;<b id="rPat" style="color:var(--accent)"></b></span>
      <span>spiking now&nbsp;<b id="rSpk">0</b></span>
      <span>global inhibition&nbsp;<b id="rInh" style="color:var(--inh)">0.0</b></span>
      <span>leader &nu;&nbsp;<b id="rNu">0.000</b>/0.2</span>
      <span id="rEvt" style="color:var(--good);font-weight:700"></span>
    </div>
    <div class="legend">
      <span><i style="background:#1f6feb"></i>excitatory neuron (fill = membrane)</span>
      <span><i style="background:#ffffff"></i>spike</span>
      <span><i style="background:#3fb950"></i>retired / consolidated</span>
      <span><i style="background:#f85149"></i>global inhibitory interneuron (&Sigma; lateral, w=&minus;5)</span>
      <span><i style="background:#58a6ff"></i>feedforward weight</span>
      <span><i style="background:#56d364"></i>self-excitation (+2)</span>
    </div>
  </div>

  <div class="panel">
    <div class="row" style="justify-content:space-between;align-items:center">
      <h2 style="margin:0">Post-hoc analysis</h2>
      <div class="toggle">
        <button id="bDense" class="on" onclick="setMode('dense')">Dense pool</button>
        <button id="bSparse" onclick="setMode('sparse')">Sparse pool</button>
      </div>
    </div>
  </div>

  <div class="panel">
    <h2>Consolidated receptive fields (retired neurons)</h2>
    <div class="rfwrap" id="rfs"></div>
  </div>

  <div class="panel">
    <h2>Cross-response matrix &mdash; selectivity</h2>
    <div class="row" style="align-items:center;gap:28px">
      <div id="xmat"></div>
      <div class="sub" style="max-width:330px">Rows = presented line, columns = owning
        neuron. Diagonal = on-target drive, off-diagonal = leakage; a spike needs
        drive &ge; &theta;.
        <div style="margin-top:10px">on-target min <b id="onmin"></b> &middot;
          off-target max <b id="offmax"></b><br/>
          separation margin <b id="margin" style="color:var(--good)"></b></div></div>
    </div>
  </div>

  <div class="panel">
    <h2>Maturation timeline (race order &amp; latency)</h2>
    <div class="tl" id="timeline"></div>
  </div>
</main>

<script>
const META = __META__;
const MODES = __MODES__;          // {dense:{...}, sparse:{...}}
const TRACES = __TRACES__;        // recorded dense dynamics, one per block
const Wff = __WFF__;              // 9 x 64 learned feedforward (dense)
const PCOL = ['#f97583','#ffab70','#f0c674','#d2a8ff',
              '#58a6ff','#56d4dd','#7ee787','#79c0ff'];
const N = META.N, D = META.input_dim, G = META.grid, THETA = META.theta;
const NM = META.names;
let mode = 'dense';

/* ---------- canvas layout ---------- */
const cv = document.getElementById('net'), ctx = cv.getContext('2d');
const W = cv.width, H = cv.height;
const inPos = [];
const ipx = 70, ipy = H/2 - 95, istep = 58;
for(let r=0;r<G;r++)for(let c=0;c<G;c++)
  inPos.push([ipx + c*istep, ipy + r*istep]);
const nPos = [];
const nx0 = 360, ny0 = 70, ncol = 8, nstep = 58;
for(let i=0;i<N;i++) nPos.push([nx0 + (i%ncol)*nstep, ny0 + Math.floor(i/ncol)*nstep]);
const inh = [W-70, H/2];

function edgeList(all){
  const e=[]; const thr = all?0.05:0.8;
  for(let n=0;n<N;n++)for(let d=0;d<D;d++){
    const w=Wff[d][n]; if(w>thr) e.push([d,n,w]);
  }
  return e;
}
let showAll=false, EDGES=edgeList(false);

/* ---------- playback state ---------- */
let bi=0, t=0, timer=null, playing=false;
function trace(){ return TRACES[bi]; }
function maturedAt(step){
  const tr=trace(); const set=new Set(tr.matured_before);
  if(step>=tr.mature_step-1) set.add(tr.winner);
  return set;
}

function draw(){
  const tr=trace(), st=tr.steps[Math.min(t,tr.steps.length-1)];
  const pat=META.patterns[tr.pid];
  const retired=maturedAt(t);
  ctx.clearRect(0,0,W,H);

  for(const [d,n,w] of EDGES){
    const active = pat[d]>0.5 && st.s.includes(n);
    ctx.strokeStyle = active ? 'rgba(121,192,255,0.9)'
                             : `rgba(88,166,255,${0.05+0.18*Math.min(1,w/2)})`;
    ctx.lineWidth = active ? 2 : 1;
    ctx.beginPath(); ctx.moveTo(inPos[d][0],inPos[d][1]);
    ctx.lineTo(nPos[n][0],nPos[n][1]); ctx.stroke();
  }
  const inhDrive = st.nspk * Math.abs(META.w_lateral);
  const ia = Math.min(0.8, 0.04 + st.nspk*0.05);
  for(let n=0;n<N;n++){
    ctx.strokeStyle = `rgba(248,81,73,${st.nspk>0?ia:0.03})`;
    ctx.lineWidth = st.nspk>0?1.4:0.6;
    ctx.beginPath(); ctx.moveTo(inh[0],inh[1]);
    ctx.lineTo(nPos[n][0],nPos[n][1]); ctx.stroke();
  }

  for(let d=0;d<D;d++){
    const on=pat[d]>0.5;
    ctx.fillStyle = on ? PCOL[tr.pid] : '#161b22';
    ctx.strokeStyle = '#30363d';
    roundRect(inPos[d][0]-22,inPos[d][1]-22,44,44,6); ctx.fill(); ctx.stroke();
  }
  ctx.fillStyle='#8b949e'; ctx.font='12px sans-serif'; ctx.textAlign='center';
  ctx.fillText('INPUT 3×3', inPos[1][0], ipy-44);

  for(let n=0;n<N;n++){
    if(retired.has(n)) continue;
    const fire=st.s.includes(n);
    ctx.strokeStyle=`rgba(86,211,100,${fire?0.9:0.18})`;
    ctx.lineWidth=fire?2:1;
    ctx.beginPath();
    ctx.arc(nPos[n][0], nPos[n][1]-16, 7, 0.5, 5.6);
    ctx.stroke();
  }

  for(let n=0;n<N;n++){
    const v=st.vq[n]/100, fire=st.s.includes(n), ret=retired.has(n);
    const [x,y]=nPos[n];
    ctx.beginPath(); ctx.arc(x,y,14,0,6.2832);
    if(ret){
      ctx.fillStyle=PCOL[NM_idx(n,retired,tr)]||'#3fb950';
    }else{
      const a=Math.min(1,v); ctx.fillStyle=`rgba(31,111,235,${0.12+0.85*a})`;
    }
    ctx.fill();
    if(fire){ ctx.lineWidth=3; ctx.strokeStyle='#ffffff';
              ctx.beginPath(); ctx.arc(x,y,16,0,6.2832); ctx.stroke(); }
    else if(ret){ ctx.lineWidth=2; ctx.strokeStyle='#3fb950';
              ctx.beginPath(); ctx.arc(x,y,15,0,6.2832); ctx.stroke(); }
    if(ret){ ctx.fillStyle='#0d1117'; ctx.font='bold 9px sans-serif';
             ctx.fillText(sym(n,retired,tr), x, y+3); }
  }
  ctx.fillStyle='#8b949e'; ctx.font='12px sans-serif';
  ctx.fillText('L2 EXCITATORY POOL (N=64)', nx0+3*nstep, ny0-26);

  const r = 26 + Math.min(20, st.nspk*1.4);
  ctx.beginPath(); ctx.arc(inh[0],inh[1],r,0,6.2832);
  ctx.fillStyle=`rgba(248,81,73,${0.35+Math.min(0.6,st.nspk*0.05)})`; ctx.fill();
  ctx.lineWidth=2; ctx.strokeStyle='#f85149'; ctx.stroke();
  ctx.fillStyle='#fff'; ctx.font='bold 10px sans-serif';
  ctx.fillText('INH', inh[0], inh[1]+3);
  ctx.fillStyle='#8b949e'; ctx.font='12px sans-serif';
  ctx.fillText('global -5', inh[0], inh[1]+r+16);

  document.getElementById('rStep').textContent=t+1;
  document.getElementById('rTot').textContent=tr.steps.length;
  document.getElementById('rPat').textContent=tr.name;
  document.getElementById('rPat').style.color=PCOL[tr.pid];
  document.getElementById('rSpk').textContent=st.s.length;
  document.getElementById('rInh').textContent=inhDrive.toFixed(1);
  document.getElementById('rNu').textContent=st.numax.toFixed(3);
  document.getElementById('rEvt').textContent =
     (t>=tr.mature_step-1) ? `▲ neuron #${tr.winner} matured → ${tr.name} retired` : '';
  document.getElementById('scrub').value=t;
}
function ownerPid(n,tr){
  if(n===tr.winner) return tr.pid;
  for(let k=0;k<MODES.dense.owners.length;k++) if(MODES.dense.owners[k]===n) return k;
  return null;
}
function NM_idx(n,retired,tr){ const p=ownerPid(n,tr); return p; }
function sym(n,retired,tr){ const p=ownerPid(n,tr);
  return p==null?'':NM[p].replace('ROW_','R').replace('COL_','C')
                       .replace('DIAG_\\','\\').replace('DIAG_/','/'); }

function roundRect(x,y,w,h,r){ctx.beginPath();ctx.moveTo(x+r,y);
  ctx.arcTo(x+w,y,x+w,y+h,r);ctx.arcTo(x+w,y+h,x,y+h,r);
  ctx.arcTo(x,y+h,x,y,r);ctx.arcTo(x,y,x+w,y,r);ctx.closePath();}

/* ---------- controls ---------- */
function loadBlock(){ const tr=trace();
  document.getElementById('scrub').max=tr.steps.length-1; t=0; draw(); }
function step(){ const tr=trace();
  if(t<tr.steps.length-1){t++; draw();} else { pause(); } }
function play(){ if(playing)return; playing=true;
  document.getElementById('play').textContent='❚❚ Pause';
  const sp=+document.getElementById('speed').value;
  timer=setInterval(step, 1000/sp); }
function pause(){ playing=false; clearInterval(timer);
  document.getElementById('play').textContent='▶ Play'; }
document.getElementById('play').onclick=()=>playing?pause():play();
document.getElementById('stepb').onclick=()=>{pause();step();};
document.getElementById('reset').onclick=()=>{pause();t=0;draw();};
document.getElementById('speed').oninput=()=>{ if(playing){pause();play();} };
document.getElementById('scrub').oninput=e=>{pause();t=+e.target.value;draw();};
document.getElementById('alledges').onclick=e=>{ showAll=!showAll;
  EDGES=edgeList(showAll); e.target.classList.toggle('on',showAll); draw(); };
const bsel=document.getElementById('block');
TRACES.forEach((tr,k)=>{const o=document.createElement('option');
  o.value=k;o.textContent=`${k+1}. ${tr.name}`;bsel.appendChild(o);});
bsel.onchange=e=>{pause();bi=+e.target.value;loadBlock();};

/* ---------- analysis panels ---------- */
function mkGrid3(){const g=document.createElement('div');g.className='grid3';
  for(let k=0;k<9;k++)g.appendChild(document.createElement('i'));return g;}
function heat(g,vals,max,base){const c=g.querySelectorAll('i');
  vals.forEach((v,k)=>{const a=max>0?Math.min(1,v/max):0;
    c[k].style.background=base?`rgba(${base},${0.08+0.92*a})`:(v>0.5?'#58a6ff':'#0d1117');});}

function renderStats(){const d=MODES[mode];
  document.getElementById('stats').innerHTML=[
    ['8 / 8','primitives consolidated'],
    [d.neurons.filter(n=>n.mature).length,'retired neurons'],
    [d.tuned_plastic,'hot spares (tuned plastic)'],
    [d.max_mass.toFixed(1),'peak L1 mass'],
    ['+'+d.margin.toFixed(2),'selectivity margin'],
  ].map(([a,b])=>`<div class="stat"><b>${a}</b><span>${b}</span></div>`).join('');}

function renderRFs(){const d=MODES[mode];const h=document.getElementById('rfs');h.innerHTML='';
  for(let p=0;p<8;p++){const n=d.neurons[d.owners[p]];
    const card=document.createElement('div');card.className='rfcard';
    const g=mkGrid3();heat(g,n.rf,Math.max(...n.rf),'88,166,255');
    const b=document.createElement('b');b.textContent=NM[p];b.style.color=PCOL[p];
    const s=document.createElement('small');s.style.color='#8b949e';
    s.textContent=`#${n.i} · sel ${n.sel.toFixed(1)}`;
    card.append(b,g,s);h.appendChild(card);}}

function renderMatrix(){const d=MODES[mode];let h='<table class="x"><tr><th></th>';
  for(let j=0;j<8;j++)h+=`<th style="color:${PCOL[j]}">${NM[j].slice(0,2)}</th>`;h+='</tr>';
  for(let p=0;p<8;p++){h+=`<tr><th style="color:${PCOL[p]}">${NM[p].slice(0,2)}</th>`;
    for(let j=0;j<8;j++){const v=d.xresp[p][j],fire=v>=THETA,a=Math.min(1,v/6);
      const bg=fire?`rgba(63,185,80,${0.25+0.6*a})`:`rgba(210,153,34,${0.12+0.5*a})`;
      h+=`<td style="background:${bg}">${v.toFixed(2)}</td>`;}h+='</tr>';}
  h+='</table>';document.getElementById('xmat').innerHTML=h;
  document.getElementById('onmin').textContent=d.on_min.toFixed(2);
  document.getElementById('offmax').textContent=d.off_max.toFixed(2);
  document.getElementById('margin').textContent='+'+d.margin.toFixed(2);}

function renderTimeline(){const d=MODES[mode];const h=document.getElementById('timeline');h.innerHTML='';
  const mx=Math.max(...d.timeline.map(t=>t.latency));
  d.timeline.forEach(t=>{const row=document.createElement('div');row.className='bar';
    row.innerHTML=`<div class="name" style="color:${PCOL[t.pid]}">${t.name}</div>
      <div class="track"><div class="fill" style="width:${100*t.latency/mx}%;
        background:${PCOL[t.pid]}"></div></div>
      <div class="lat">#${t.winner} · ${t.latency} st</div>`;h.appendChild(row);});}

function setMode(m){mode=m;
  document.getElementById('bDense').classList.toggle('on',m==='dense');
  document.getElementById('bSparse').classList.toggle('on',m==='sparse');
  renderStats();renderRFs();renderMatrix();renderTimeline();}

loadBlock(); setMode('dense');
</script>
</body>
</html>
"""


def main():
    pats, names = build_line_patterns()
    dense = train_capture(sparse=False, record=True)
    sparse = train_capture(sparse=True, record=False)
    cfg = SNNConfig(input_dim=pats.shape[1])

    meta = dict(N=cfg.n_neurons, input_dim=cfg.input_dim, grid=3,
                theta=cfg.theta_rest, w_lateral=cfg.w_lateral,
                w_self=cfg.w_self, w_total=cfg.w_total,
                names=names, patterns=pats.tolist())

    modes = {"dense": {k: v for k, v in dense.items()
                       if k not in ("traces", "W")},
             "sparse": {k: v for k, v in sparse.items() if k != "W"}}

    html = (HTML
            .replace("__META__", json.dumps(meta))
            .replace("__MODES__", json.dumps(modes))
            .replace("__TRACES__", json.dumps(dense["traces"]))
            .replace("__WFF__", json.dumps(dense["W"])))

    out = "snn_visualization.html"
    with open(out, "w") as f:
        f.write(html)

    nsteps = sum(len(t["steps"]) for t in dense["traces"])
    print(f"wrote {out}  ({len(html)//1024} KB, {nsteps} recorded dynamics steps)")
    for tag, d in (("DENSE", dense), ("SPARSE", sparse)):
        print(f"  [{tag:>6}] retired=8/8  hot_spares={d['tuned_plastic']:2d}  "
              f"peak_mass={d['max_mass']:5.2f}  margin=+{d['margin']:.2f}")


if __name__ == "__main__":
    main()
