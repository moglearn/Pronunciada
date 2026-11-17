"""Render the overview film (docs/assets/demo.mp4, about 45 s, 1920x1080, 60 frames per second).

    pip install playwright          # drives the installed Google Chrome
    python scripts/graphics/make_video.py [--sheet out.jpg]

The film is one HTML page whose whole state is a function of time: every motion is a paused
Web Animation, and `seek(t)` moves all of them to second t. Each frame is seeked and
screenshotted, so the motion is exact however slow the machine is.

It follows one recording (site/samples/en-learner-1) from the sentence to its word scores and
the sounds of the word that was flagged, then shows the ten languages, a Mandarin phrase with
its tones, the speed, the results against Azure and ZIPA alone, and the logo. Numbers are read
from results/SCOREBOARD.md. The recording is the only sound. A copy of the film and one frame
of it (the cover image) go to site/assets/. --sheet writes a contact sheet of still frames.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from playwright.sync_api import sync_playwright

from common import ROOT, font_css
from make_charts import BOARD, english, table, zipa
from make_logo import SHAPE, wordmark

OUT = ROOT / "docs/assets/demo.mp4"
SITE = ROOT / "site/assets"
SAMPLES = ROOT / "site/samples"
LENGTH, FPS = 44.8, 60
VOICES = {"en-learner-1": 0.6}        # second at which each recording starts
COVER_AT = 5.55                      # frame used as the cover image on the site


def clip(key: str) -> dict:
    """What the film needs from one sample: the stretch of audio to play and the timed pieces of its text."""
    d = json.loads((SAMPLES / f"{key}.json").read_text(encoding="utf-8"))["result"]
    words = d["words"]
    a0, a1 = max(0.0, words[0]["start"] - 0.2), words[-1]["end"] + 0.2
    return {"a0": a0, "dur": a1 - a0, "result": d,
            "words": [{"text": w["text"], "prob": w["prob"], "start": w["start"], "end": w["end"]} for w in words]}


def data() -> dict:
    one, zh = clip("en-learner-1"), clip("zh-learner-1")
    audio, sr = sf.read(str(SAMPLES / "en-learner-1.mp3"), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    n = 147
    edges = np.linspace(one["a0"], one["a0"] + one["dur"], n + 1)
    level = np.array([np.abs(mono[int(a * sr):max(int(b * sr), int(a * sr) + 1)]).max() for a, b in zip(edges, edges[1:])])
    level = (level / level.max()) ** 0.7
    low = next(w for w in one["result"]["words"] if w["prob"] < 0.5)
    counts = [[int(x) for x in row[1:7]] for row in table("By language:")]
    m = next(re.search(r"over (\d+) rows: significantly ahead (\d+)", line) for line in BOARD if line.startswith("Pronunciada vs gop"))
    ph = next(r for r in english() if r["label"] == "Phoneme")
    return {
        "at": VOICES, "a0": one["a0"], "dur": one["dur"], "levels": [round(float(v), 3) for v in level],
        "words": one["words"],
        "low": {"text": low["text"], "start": low["start"], "end": low["end"],
                "phones": [{"ipa": p["ipa"], "prob": p["prob"], "heard": p["heard"]} for p in low["phones"]]},
        "tip": one["result"]["tips"][0]["text"],
        "zh": {"a0": zh["a0"], "dur": zh["dur"],
               "syl": [{"text": s["text"], "start": s["start"], "exp": s["tone_expected"], "heard": s["tone_heard"], "low": s["tone_prob"] < 0.5}
                       for w in zh["result"]["words"] for s in w["syllables"]]},
        "tally": {"n": sum(c[0] for c in counts), "better": sum(c[1] for c in counts),
                  "same": sum(c[2] + c[3] + c[4] for c in counts), "behind": sum(c[5] for c in counts)},
        "bars": {"ours": ph["ours"][0], "azure": ph["azure"][0], "zipa": zipa()[0]["azure"][0]},
        "zipa": {"n": int(m.group(1)), "ahead": int(m.group(2))},
        "shape": list(SHAPE), "name": wordmark("#000000", with_mark=False),
    }


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><style>__FONTS__
*{box-sizing:border-box;margin:0;padding:0}
html,body{width:1280px;height:720px;overflow:hidden;background:#fdfcfc;color:#0b0b0c;font-family:'Geist','Inter','Noto Sans CJK SC','PingFang SC','Hiragino Sans','Apple SD Gothic Neo',sans-serif;-webkit-font-smoothing:antialiased}
.glow{position:absolute;width:1100px;height:1100px;border-radius:50%}
#g1{left:560px;top:-560px;background:radial-gradient(closest-side,rgba(29,78,216,.13),rgba(29,78,216,0))}
#g2{left:-420px;top:260px;background:radial-gradient(closest-side,rgba(14,165,160,.12),rgba(14,165,160,0))}
.sc{position:absolute;inset:0;opacity:0}
.c{position:absolute;left:0;right:0;text-align:center}
.mono{font-family:'Geist Mono',monospace}
.soft{color:#6b6661}
.grad{background:linear-gradient(90deg,#1d4ed8,#0ea5a0);-webkit-background-clip:text;background-clip:text;color:transparent;-webkit-text-fill-color:transparent}
.cap{font-size:26px;font-weight:400;letter-spacing:-.01em;color:#6b6661}
.h{font-size:64px;font-weight:300;letter-spacing:-.035em;line-height:1.08}
.line{font-size:58px;font-weight:300;letter-spacing:-.03em;white-space:nowrap}
.label{font:400 17px 'Geist Mono',monospace;letter-spacing:.06em;color:#8c8782}
.tip{font-size:34px;font-weight:300;letter-spacing:-.015em}
/* the first recording */
.wave{position:absolute;left:200px;width:880px;height:130px}
.wave div{position:absolute;inset:0}
.wave i{position:absolute;width:3px;border-radius:2px;background:#0b0b0c}
.wd{display:inline-block;margin:0 8px;text-align:center}
.wd .t{display:block;position:relative}
.wd .t .g{position:absolute;left:0;top:0;opacity:0}
.wd .p{display:block;margin-top:14px;font:400 23px 'Geist Mono',monospace;letter-spacing:0;color:#8c8782;opacity:0}
/* one word, sound by sound */
#bw{position:absolute;font-size:150px;font-weight:300;letter-spacing:-.045em;line-height:1;white-space:nowrap;transform-origin:0 0;opacity:0}
#bw .seg{display:inline-block;position:relative}
#bw .seg > span{display:block}
#bw .so{position:absolute;left:0;top:0;opacity:0;white-space:nowrap}
#bw .seg .sw{display:inline-block;position:relative}
#bw .seg .sw span{display:block}
#bw .seg .sw .new{position:absolute;left:-.2em;right:-.2em;top:0;text-align:center;opacity:0}
.pn{position:absolute;width:200px;text-align:center;font:400 30px 'Geist Mono',monospace;opacity:0}
.px{position:absolute;width:200px;text-align:center;font-size:22px;color:#6b6661;opacity:0}
#dot{position:absolute;width:26px;height:26px;border-radius:50%;background:linear-gradient(135deg,#1d4ed8,#0ea5a0);opacity:0}
/* tones and stress */
#zh{top:286px;font-size:120px;font-weight:300;white-space:nowrap}
#zh span{display:inline-block;position:relative;width:150px}
#zh span .g{position:absolute;left:0;right:0;top:0;opacity:0}
.fw{position:relative;display:inline-block;margin:0 8px;color:#cfcac4}
.fw .g{position:absolute;left:0;top:0;opacity:0}
/* languages and speed */
#langs{position:absolute;left:0;right:0;top:352px;display:grid;grid-template-columns:repeat(5,max-content);justify-content:center;justify-items:center;gap:26px 60px;font-size:48px;font-weight:300;letter-spacing:-.02em;color:#cfcac4}
#langs span{position:relative}#langs span b{position:absolute;left:0;top:0;font-weight:300;color:#0b0b0c;opacity:0}
#ms{top:226px;font-size:230px;font-weight:300;letter-spacing:-.05em;line-height:1;font-variant-numeric:tabular-nums}
#ms small{font-size:76px;letter-spacing:-.03em;color:#6b6661;margin-left:14px}
/* results */
#R1,#R2{top:18px}#Z{top:14px}
#dots{position:absolute;left:361px;top:214px;width:558px;height:238px}
#dots i{position:absolute;width:26px;height:26px;border-radius:50%;background:#e9e5e1;transform:scale(0)}
#legend{top:484px;font-size:25px;color:#6b6661}
#legend b{font-weight:500;color:#0b0b0c}#legend span{opacity:0;margin:0 16px;display:inline-block}
.row{position:absolute;left:210px;right:210px;height:70px;opacity:0}
.row .who{position:absolute;left:0;top:14px;font-size:27px;letter-spacing:-.015em}
.row .who small{display:block;font-size:19px;color:#6b6661;letter-spacing:0;margin-top:2px}
.row .bar{position:absolute;left:250px;top:24px;height:18px;border-radius:9px;background:#dcd7d2;width:0}
.row.own .bar{background:linear-gradient(90deg,#1d4ed8,#0ea5a0)}
.row .num{position:absolute;right:0;top:0;font-size:56px;font-weight:300;letter-spacing:-.04em;font-variant-numeric:tabular-nums;color:#8c8782}
.row.own .num{color:#0b0b0c}
/* close */
#mark i{position:absolute;background:#0b0b0c;opacity:0}
#mark .dotg{background:linear-gradient(135deg,#1d4ed8,#0ea5a0)}
#name{position:absolute;opacity:0}#name svg{height:100%;display:block}
#white{position:absolute;inset:0;background:#fdfcfc;opacity:0}
</style></head><body>
<div class="glow" id="g1"></div><div class="glow" id="g2"></div>

<div class="sc" id="A"><div class="wave" id="wave" style="top:236px"></div><div class="c line" id="sent" style="top:424px"></div>
  <div class="c cap" id="capA" style="top:636px;opacity:0">Each word gets the probability that a human rater would accept it.</div></div>

<div class="sc" id="B"><div id="bw"></div><div id="dot"></div><div id="labels"></div>
  <div class="c label" id="tipl" style="top:538px;opacity:0">COACHING TIP</div><div class="c tip" id="tip" style="top:572px;opacity:0"></div>
  <div class="c cap" id="capB" style="top:132px;opacity:0">Each sound gets one too.</div></div>

<div class="sc" id="T"><div class="c h" style="top:150px">Tones, stress and length.</div><div class="c" id="zh"></div>
  <div class="c cap" id="zhn" style="top:478px;opacity:0"></div></div>

<div class="sc" id="L"><div class="c h" style="top:200px">Ten languages.</div><div id="langs"></div></div>
<div class="sc" id="M"><div class="c" id="ms"><span>0</span><small>ms</small></div>
  <div class="c cap" style="top:500px">per sentence on a laptop CPU · no cloud · a 67 MB model</div></div>

<div class="sc" id="R1"><div class="c cap" style="top:124px">__N__ public comparisons with Microsoft Azure Pronunciation Assessment</div><div id="dots"></div>
  <div class="c" id="legend"></div><div class="c h" id="r1h" style="top:560px;font-size:44px;opacity:0"></div></div>
<div class="sc" id="R2"><div class="c cap" style="top:124px">Agreement with expert raters, English learners</div>
  <div class="row own" style="top:232px"><div class="who">Pronunciada</div><div class="bar"></div><div class="num"></div></div>
  <div class="row" style="top:322px"><div class="who">Azure</div><div class="bar"></div><div class="num"></div></div>
  <div class="row" style="top:412px"><div class="who">ZIPA alone<small>the open model it is built on</small></div><div class="bar"></div><div class="num"></div></div>
  <div class="c h" id="r2h" style="top:560px;font-size:40px;opacity:0"></div></div>

<div class="sc" id="Z"><div class="wave" id="wave2" style="top:236px;opacity:0"></div><div id="mark"></div><div id="name"></div>
  <div class="c" id="zh1" style="top:402px;font-size:32px;font-weight:300;letter-spacing:-.02em;opacity:0">On-device pronunciation assessment with cloud-level accuracy.</div>
  <div class="c mono soft" id="zh2" style="top:468px;font-size:19px;opacity:0">Open source · github.com/moglearn/pronunciada</div></div>
<div id="white"></div>

<script>
const D=__DATA__, LEN=__LEN__, anims=[], ticks=[], $=s=>document.querySelector(s), EASE='cubic-bezier(.2,.7,.2,1)', INOUT='cubic-bezier(.65,0,.35,1)';
function A(el,kf,t0,d,ease){const a=el.animate(kf,{delay:t0*1000,duration:d*1000,fill:'both',easing:ease||EASE});a.pause();anims.push(a);return a;}
function fade(el,t0,d,rise){A(el,[{opacity:0,transform:`translateY(${rise==null?10:rise}px)`},{opacity:1,transform:'none'}],t0,d||.5);}
function life(el,tin,tout,din,dout){din=din||.5;dout=dout||.4;const T=tout+dout-tin;
  A(el,[{opacity:0,offset:0,easing:EASE},{opacity:1,offset:din/T},{opacity:1,offset:(tout-tin)/T,easing:'ease-in'},{opacity:0,offset:1}],tin,T,'linear');}
function count(el,t0,d,to,dec){ticks.push(t=>{const k=Math.min(1,Math.max(0,(t-t0)/d)),e=1-Math.pow(1-k,3);el.textContent=(to*e).toFixed(dec||0);});}
const mix=k=>{k=Math.min(1,Math.max(0,k));const a=[29,78,216],b=[14,165,160];return 'rgb('+a.map((v,i)=>Math.round(v+(b[i]-v)*k)).join(',')+')';};
const el=(tag,cls,html)=>{const e=document.createElement(tag);if(cls)e.className=cls;if(html!=null)e.innerHTML=html;return e;};
const N=D.levels.length;
function bars(host,color){D.levels.forEach((v,i)=>{const h=Math.max(4,2*Math.round(v*62)),b=el('i');b.style.cssText=`left:${i*6}px;height:${h}px;top:${65-h/2}px`+(color?`;background:${color(i)}`:'');host.appendChild(b);});return host;}

function setup(){
A($('#g1'),[{transform:'translate(0,0)'},{transform:'translate(-150px,110px)'}],0,LEN,'linear');
A($('#g2'),[{transform:'translate(0,0)'},{transform:'translate(170px,-90px)'}],0,LEN,'linear');

/* A: the first recording, its words, their scores */
const V1=D.at['en-learner-1'], HI=3.9;
life($('#A'),0,5.75,.4,.35);
const wave=$('#wave'), faint=bars(el('div')), ink=bars(el('div')), tint=bars(el('div'),i=>mix((i/N*D.dur+D.a0-D.low.start)/(D.low.end-D.low.start)));
faint.style.opacity=.16;wave.append(faint,ink,tint);
A(ink,[{clipPath:'inset(0 100% 0 0)'},{clipPath:'inset(0 0% 0 0)'}],V1,D.dur,'linear');
const xa=(D.low.start-D.a0)/D.dur*100, xb=(D.low.end-D.a0)/D.dur*100;
tint.style.clipPath=`inset(0 ${(100-xb).toFixed(2)}% 0 ${xa.toFixed(2)}%)`;A(tint,[{opacity:0},{opacity:1}],HI,.6);
let flagged=null;
D.words.forEach((w,i)=>{const low=w.prob<.5, e=el('span','wd',`<span class="t"><span class="k">${w.text}</span>${low?`<span class="g grad">${w.text}</span>`:''}</span><span class="p">${w.prob.toFixed(2)}</span>`);
  $('#sent').appendChild(e);
  A(e,[{opacity:0,transform:'translateY(8px)'},{opacity:1,transform:'none'}],V1+(w.start-D.a0)-.04,.32);
  fade(e.querySelector('.p'),3.0+i*.06,.45,6);
  if(low){flagged=e.querySelector('.t');A(e.querySelector('.k'),[{opacity:1},{opacity:0}],HI,.6);A(e.querySelector('.g'),[{opacity:0},{opacity:1}],HI,.6);
          A(e.querySelector('.p'),[{color:'#8c8782'},{color:'#1d4ed8'}],HI,.6);}});
life($('#capA'),3.2,5.0,.5,.35);

/* B: the flagged word moves to the centre and its letters part into sounds */
const IN=5.75, SPREAD=6.9, P=D.low.phones;
life($('#B'),IN,12.8,.01,.4);
const parts=D.low.text.match(/^(f)(l)(oa)(t)$/)?['f','l','oa','t']:D.low.text.split(''), bw=$('#bw');
const segs=parts.map((ch,k)=>{const ipa=P[k]?P[k].ipa:ch;let c=0;while(c<ch.length&&ch[c]===ipa[c])c++;
  const solid=ipa===ch?ch:`${ch.slice(0,c)}<span class="sw"><span class="old">${ch.slice(c)}</span><span class="new">${ipa.slice(c)}</span></span>`;
  const s=el('span','seg',`<span class="gr">${ch}</span><span class="so">${solid}</span>`);bw.appendChild(s);return s;});
const W=bw.offsetWidth, LEFT=640-W/2, TOP=236, xs=segs.map(s=>s.offsetLeft), ws=segs.map(s=>s.offsetWidth);
bw.style.left=LEFT+'px';bw.style.top=TOP+'px';
segs.forEach((s,k)=>{const g=s.querySelector('.gr');g.style.cssText=`background:linear-gradient(90deg,#1d4ed8,#0ea5a0) ${-xs[k]}px 0/${W}px 100% no-repeat;-webkit-background-clip:text;background-clip:text;color:transparent;-webkit-text-fill-color:transparent`;});
const r=flagged.getBoundingClientRect(), fs=parseFloat(getComputedStyle(flagged).fontSize), s0=fs/parseFloat(getComputedStyle(bw).fontSize), top0=r.top+(r.height-fs)/2;
A(bw,[{opacity:0},{opacity:1}],IN,.05,'linear');A(flagged,[{opacity:1},{opacity:0}],IN,.05,'linear');
A(bw,[{transform:`translate(${r.left-LEFT}px,${top0-TOP}px) scale(${s0})`},{transform:'none'}],IN,.95,INOUT);
const cx=k=>640+(k-(parts.length-1)/2)*200, labels=$('#labels');
segs.forEach((s,k)=>{const p=P[k], dx=cx(k)-(LEFT+xs[k]+ws[k]/2), swap=s.querySelector('.sw');
  A(s,[{transform:'translateX(0)'},{transform:`translateX(${dx}px)`}],SPREAD,.85,INOUT);
  A(s.querySelector('.gr'),[{opacity:1},{opacity:0}],SPREAD,.85,'linear');
  A(s.querySelector('.so'),[{opacity:0},{opacity:1}],SPREAD,.85,'linear');
  if(swap){A(swap.querySelector('.old'),[{opacity:1,transform:'translateY(0)'},{opacity:0,transform:'translateY(-14px)'}],SPREAD+.95,.3,'ease-in');
           A(swap.querySelector('.new'),[{opacity:0,transform:'translateY(14px)'},{opacity:1,transform:'none'}],SPREAD+1.15,.35);}
  if(!p) return;
  const n=el('div','pn',p.prob.toFixed(2));n.style.cssText=`left:${cx(k)-100}px;top:${TOP+176}px;color:${p.prob<.5?'#1d4ed8':'#0b0b0c'}`;labels.appendChild(n);fade(n,8.45+k*.09,.45,6);
  const note=p.heard==null?'':(p.heard===''?'not heard':'heard as '+p.heard);
  if(note){const x=el('div','px',note);x.style.cssText=`left:${cx(k)-100}px;top:${TOP+222}px`;labels.appendChild(x);fade(x,9.6+k*.08,.45,5);}
  if(p.heard===''){const d=$('#dot');d.style.left=(cx(k)-13)+'px';d.style.top=(TOP+74)+'px';
    A(s,[{opacity:1},{opacity:0}],9.0,.4,'ease-in').effect.updateTiming({fill:'forwards'});
    A(d,[{opacity:0,transform:'scale(.3)'},{opacity:1,transform:'scale(1)'}],9.25,.5);}});
life($('#capB'),8.3,12.3,.5,.35);
fade($('#tipl'),10.2,.45,5);$('#tip').textContent=D.tip;
A($('#tip'),[{opacity:0,clipPath:'inset(0 100% 0 0)'},{opacity:1,clipPath:'inset(0 0% 0 0)'}],10.35,1.0,'cubic-bezier(.4,0,.2,1)');

/* L, T, M: languages, tones, speed */
life($('#L'),13.2,16.9);life($('#T'),17.3,21.4);life($('#M'),21.8,25.0);
['English','中文','日本語','Español','Français','Deutsch','Italiano','Português','Русский','한국어'].forEach((t,i)=>{const e=el('span','',t+'<b>'+t+'</b>');
  $('#langs').appendChild(e);A(e.querySelector('b'),[{opacity:0},{opacity:1}],13.7+i*.2,.4);});
D.zh.syl.forEach(s=>{const e=el('span','',s.text+(s.low?`<span class="g grad">${s.text}</span>`:''));$('#zh').appendChild(e);
  if(s.low) A(e.querySelector('.g'),[{opacity:0},{opacity:1}],18.3,.5);});
$('#zhn').innerHTML=D.zh.syl.filter(s=>s.low).map(s=>`${s.text}  tone ${s.exp} expected, tone ${s.heard} heard`).join('   ·   ');fade($('#zhn'),18.7,.5,6);
count($('#ms span'),22.0,1.0,50,0);

/* R: results */
const R0=25.4;
life($('#R1'),R0,R0+6.0);life($('#R2'),R0+6.4,R0+12.1);
const T=D.tally, dots=$('#dots');
for(let i=0;i<T.n;i++){const e=el('i'), col=i%14,row=Math.floor(i/14);e.style.cssText=`left:${col*41}px;top:${row*42.4}px`;dots.appendChild(e);
  A(e,[{transform:'scale(0)'},{transform:'scale(1)'}],R0+.2+(col+row)*.035,.4);
  if(i<T.better) A(e,[{backgroundColor:'#e9e5e1'},{backgroundColor:mix(col/13)}],R0+1.2+i*.022,.4);
  else if(i>=T.n-T.behind) A(e,[{backgroundColor:'#e9e5e1'},{backgroundColor:'#8c8782'}],R0+3.1+(i-(T.n-T.behind))*.04,.4);}
$('#legend').innerHTML=`<span><b>${T.better}</b> significantly better</span><span><b>${T.same}</b> no significant difference</span><span><b>${T.behind}</b> behind</span>`;
[...$('#legend').children].forEach((e,i)=>fade(e,R0+[1.4,2.4,3.2][i],.45,5));
$('#r1h').innerHTML=`Level with or ahead in <span class="grad">${T.better+T.same} of ${T.n}</span>.`;fade($('#r1h'),R0+3.8,.6);
[['ours',R0+6.8],['azure',R0+7.5],['zipa',R0+8.5]].forEach(([k,t],i)=>{const r=document.querySelectorAll('.row')[i], v=D.bars[k];
  A(r,[{opacity:0},{opacity:1}],t-.25,.45);A(r.querySelector('.bar'),[{width:'0px'},{width:(v*640)+'px'}],t,.9);count(r.querySelector('.num'),t,.9,v,2);});
$('#r2h').innerHTML=`Significantly better than ZIPA alone on <span class="grad">${D.zipa.ahead} of ${D.zipa.n}</span> tests.`;fade($('#r2h'),R0+9.6,.6);

/* Z: the waveform becomes the logo */
const Z0=R0+12.5;
life($('#Z'),Z0,LEN,.4,.1);
const w2=bars($('#wave2'));
A(w2,[{opacity:0,offset:0},{opacity:1,offset:.35},{opacity:1,offset:.55},{opacity:0,offset:1}],Z0,1.5,'linear');
const H=81.6, MH=H*.8/1.02, bw2=MH*.105/.8, gap=MH*.075/.8, mw=5*bw2+4*gap, vb=D.name.match(/viewBox="0 -?[\d.]+ ([\d.]+) ([\d.]+)"/), nw=H*vb[1]/vb[2], lock=mw+H*.3/1.02+nw, X0=640-lock/2, Y0=262, CY=Y0+H*.412;
const name=$('#name');name.innerHTML=D.name;name.style.cssText=`left:${X0+mw+H*.3/1.02}px;top:${Y0}px;height:${H}px`;
D.shape.forEach((s,i)=>{const idx=Math.round([.14,.31,.5,.69,.86][i]*(N-1)), h0=Math.max(8,D.levels[idx]*124), x0=200+idx*6, isDot=s==null;
  const fh=isDot?bw2*1.32:s*MH, fw=isDot?bw2*1.32:bw2, fx=X0+i*(bw2+gap)+(isDot?(bw2-fw)/2:0);
  const kf=[{left:x0+'px',top:(301-h0/2)+'px',width:'3px',height:h0+'px',borderRadius:'2px'},{left:fx+'px',top:(CY-fh/2)+'px',width:fw+'px',height:fh+'px',borderRadius:(fw/2)+'px'}];
  const e=el('i');$('#mark').appendChild(e);A(e,[{opacity:0},{opacity:1}],Z0+.7,.4);A(e,kf,Z0+1.3,1.05,INOUT);
  if(isDot){const g=el('i','dotg');$('#mark').appendChild(g);A(g,kf,Z0+1.3,1.05,INOUT);A(g,[{opacity:0},{opacity:1}],Z0+2.2,.45);
            A(e,[{opacity:1},{opacity:0}],Z0+2.3,.45).effect.updateTiming({fill:'forwards'});}});
A(name,[{opacity:0,transform:'translateX(-12px)'},{opacity:1,transform:'none'}],Z0+2.4,.7);
fade($('#zh1'),Z0+3.1,.7,8);fade($('#zh2'),Z0+3.7,.7,6);
A($('#white'),[{opacity:0},{opacity:1}],LEN-1.1,.9,'ease-in-out');
window.seek=t=>{anims.forEach(a=>a.currentTime=t*1000);ticks.forEach(f=>f(t));};seek(0);
}
Promise.all(["300 20px Geist","500 20px Geist","400 20px 'Geist Mono'","400 20px Inter"].map(f=>document.fonts.load(f))).then(()=>document.fonts.ready).then(setup);
</script></body></html>"""


def page_html() -> str:
    d = data()
    return (PAGE.replace("__FONTS__", font_css()).replace("__DATA__", json.dumps(d, ensure_ascii=False))
            .replace("__LEN__", str(LENGTH)).replace("__N__", str(d["tally"]["n"])))


def sound(path: Path) -> None:
    """The three recordings at their places in the film, each trimmed to its speech and levelled."""
    sr = 44100
    mix = np.zeros(int(LENGTH * sr))
    for key, at in VOICES.items():
        c = clip(key)
        voice, vsr = sf.read(str(SAMPLES / f"{key}.mp3"), dtype="float32", always_2d=True)
        v = voice.mean(axis=1)[int(c["a0"] * vsr):int((c["a0"] + c["dur"]) * vsr)]
        v = np.interp(np.arange(int(len(v) * sr / vsr)) * vsr / sr, np.arange(len(v)), v)
        ramp = int(0.05 * sr)
        v[:ramp] *= np.linspace(0, 1, ramp)
        v[-3 * ramp:] *= np.linspace(1, 0, 3 * ramp)
        start = int(at * sr)
        mix[start:start + len(v)] += 0.7 * v / np.abs(v).max()
    sf.write(str(path), np.stack([mix, mix], axis=1), sr)


def main(sheet: str | None) -> None:
    with tempfile.TemporaryDirectory() as d, sync_playwright() as play:
        tmp = Path(d)
        (tmp / "film.html").write_text(page_html(), encoding="utf-8")
        browser = play.chromium.launch(channel="chrome", headless=True)
        page = browser.new_context(viewport={"width": 1280, "height": 720}, device_scale_factor=1.5).new_page()
        page.goto((tmp / "film.html").as_uri())
        page.wait_for_function("typeof window.seek === 'function'")
        if sheet:
            times = [1.7, 3.6, 5.5, 5.95, 6.4, 7.3, 7.8, 8.15, 9.3, 11.9, 15.4, 20.0, 23.6, 28.4, 30.8, 34.4, 36.9, 39.0, 39.8, 42.8]
            for i, t in enumerate(times):
                page.evaluate(f"seek({t})")
                page.screenshot(path=str(tmp / f"s{i:02d}.png"))
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp / "s%02d.png"), "-vf", "scale=480:-1,tile=4x5", sheet], check=True)
            browser.close()
            return
        sound(tmp / "sound.wav")
        enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(FPS), "-c:v", "mjpeg", "-i", "-",
                                "-i", str(tmp / "sound.wav"), "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
                                "-vf", "scale=1920:1080:flags=lanczos", "-movflags", "+faststart", "-c:a", "aac", "-b:a", "160k", "-shortest", str(OUT)],
                               stdin=subprocess.PIPE)
        for k in range(int(LENGTH * FPS)):
            page.evaluate(f"seek({k / FPS})")
            enc.stdin.write(page.screenshot(type="jpeg", quality=95))
        enc.stdin.close()
        enc.wait()
        browser.close()
    shutil.copy2(OUT, SITE / "demo.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(COVER_AT), "-i", str(OUT), "-frames:v", "1", "-vf", "scale=1616:-2",
                    "-q:v", "2", str(SITE / "cover.jpg")], check=True)
    print(f"wrote {OUT.relative_to(ROOT)} ({LENGTH:.1f} s, {OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--sheet" else None)
