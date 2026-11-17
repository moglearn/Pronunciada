"""Render the overview film (docs/assets/demo.mp4, about 52 s, 1920x1080, 60 frames per second).

    pip install playwright          # drives the installed Google Chrome
    python scripts/graphics/make_video.py [--sheet out.jpg]

The film is one HTML page whose whole state is a function of time: every motion is a paused
Web Animation, and `seek(t)` moves all of them to second t. Each frame is seeked and
screenshotted, so the motion is exact however slow the machine is. The story follows one
recording (site/samples/en-learner-1): its words and scores, the sounds of the word that was
flagged, three features, the results against Azure and ZIPA alone, and the logo. Numbers are
read from results/SCOREBOARD.md. Sound is that recording plus a quiet synthesized tone. A copy of the film and one frame of it (the cover
image) go to site/assets/.
--sheet writes a contact sheet of still frames instead of the film.
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
SAMPLE = ROOT / "site/samples/en-learner-1"
LENGTH, FPS = 52.0, 60
VOICE_AT = 0.9                       # second at which the recording starts
COVER_AT = 7.68                      # frame used as the cover image on the site


def data() -> dict:
    d = json.loads(SAMPLE.with_suffix(".json").read_text(encoding="utf-8"))["result"]
    words = d["words"]
    audio, sr = sf.read(str(SAMPLE.with_suffix(".mp3")), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    a0, a1 = words[0]["start"] - 0.2, words[-1]["end"] + 0.2
    n = 147
    edges = np.linspace(a0, a1, n + 1)
    level = np.array([np.abs(mono[int(a * sr):max(int(b * sr), int(a * sr) + 1)]).max() for a, b in zip(edges, edges[1:])])
    level = (level / level.max()) ** 0.7
    low = next(w for w in words if w["prob"] < 0.5)
    zh = json.loads((ROOT / "site/samples/zh-learner-1.json").read_text(encoding="utf-8"))["result"]
    counts = [[int(x) for x in row[1:7]] for row in table("By language:")]
    m = next(re.search(r"over (\d+) rows: significantly ahead (\d+)", line) for line in BOARD if line.startswith("Pronunciada vs gop"))
    ph = next(r for r in english() if r["label"] == "Phoneme")
    return {
        "a0": a0, "dur": a1 - a0, "levels": [round(float(v), 3) for v in level], "times": [round(float(t), 3) for t in edges[:-1]],
        "words": [{"text": w["text"], "prob": w["prob"], "start": w["start"], "end": w["end"]} for w in words],
        "low": {"text": low["text"], "prob": low["prob"], "start": low["start"], "end": low["end"],
                "phones": [{"ipa": p["ipa"], "prob": p["prob"], "heard": p["heard"]} for p in low["phones"]]},
        "tip": d["tips"][0]["text"], "sentence": d["prob"],
        "zh": [{"text": s["text"], "exp": s["tone_expected"], "heard": s["tone_heard"], "low": s["tone_prob"] < 0.5}
               for w in zh["words"] for s in w["syllables"]],
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
#A{top:52px}#B{top:58px}#F1,#F2,#F3{top:34px}#R1,#R2{top:18px}#Z{top:14px}
.abs{position:absolute}
.c{position:absolute;left:0;right:0;text-align:center}
.mono{font-family:'Geist Mono',monospace}
.soft{color:#6b6661}
.grad{background:linear-gradient(90deg,#1d4ed8,#0ea5a0);-webkit-background-clip:text;background-clip:text;color:transparent;-webkit-text-fill-color:transparent}
.cap{font-size:26px;font-weight:400;letter-spacing:-.01em;color:#6b6661}
.h{font-size:64px;font-weight:300;letter-spacing:-.035em;line-height:1.08}
/* example */
#wave{position:absolute;left:200px;top:196px;width:880px;height:130px}
#wave i,#wave2 i{position:absolute;width:3px;border-radius:2px;background:#0b0b0c;transform-origin:50% 50%}
#sent{top:386px;font-size:58px;font-weight:300;letter-spacing:-.03em;white-space:nowrap}
.wd{display:inline-block;margin:0 8px;text-align:center;opacity:0}
.wd .t{display:block;position:relative}
.wd .t .g{position:absolute;left:0;top:0;opacity:0}
.wd .p{display:block;margin-top:14px;font:400 22px 'Geist Mono',monospace;letter-spacing:0;color:#8c8782;opacity:0}
/* sounds */
#big{top:196px;font-size:150px;font-weight:300;letter-spacing:-.045em;line-height:1}
.ph{position:absolute;top:186px;width:190px;text-align:center;opacity:0}
.ph .i{display:block;font-size:124px;font-weight:300;line-height:1.1;letter-spacing:-.03em}
.ph .n{display:block;margin-top:14px;font:400 30px 'Geist Mono',monospace;opacity:0}
.ph .x{display:block;margin-top:10px;font-size:22px;color:#6b6661;opacity:0}
.ph .dot{position:absolute;left:83px;top:64px;width:24px;height:24px;border-radius:50%;background:linear-gradient(135deg,#1d4ed8,#0ea5a0);opacity:0}
#tipl{top:494px;font:400 17px 'Geist Mono',monospace;letter-spacing:.06em;color:#8c8782}
#tip{top:530px;font-size:34px;font-weight:300;letter-spacing:-.015em}
/* features */
#langs{position:absolute;left:0;right:0;top:318px;display:grid;grid-template-columns:repeat(5,max-content);justify-content:center;justify-items:center;gap:26px 60px;font-size:48px;font-weight:300;letter-spacing:-.02em;color:#cfcac4}
#langs span{position:relative}#langs span b{position:absolute;left:0;top:0;font-weight:300;color:#0b0b0c;opacity:0}
#zh{top:300px;font-size:120px;font-weight:300;letter-spacing:.04em;white-space:nowrap}
#zh span{display:inline-block;position:relative;width:150px}
#zh span .g{position:absolute;left:0;right:0;top:0;opacity:0}
#ms{top:262px;font-size:210px;font-weight:300;letter-spacing:-.05em;line-height:1;font-variant-numeric:tabular-nums}
#ms small{font-size:70px;letter-spacing:-.03em;color:#6b6661;margin-left:14px}
/* results */
#dots{position:absolute;left:361px;top:214px;width:558px;height:238px}
#dots i{position:absolute;width:26px;height:26px;border-radius:50%;background:#e9e5e1;transform:scale(0)}
#legend{top:484px;font-size:25px;color:#6b6661;word-spacing:1px}
#legend b{font-weight:500;color:#0b0b0c}#legend span{opacity:0;margin:0 16px;display:inline-block}
.row{position:absolute;left:210px;right:210px;height:70px}
.row .who{position:absolute;left:0;top:14px;font-size:27px;letter-spacing:-.015em}
.row .who small{display:block;font-size:19px;color:#6b6661;letter-spacing:0;margin-top:2px}
.row .bar{position:absolute;left:250px;top:24px;height:18px;border-radius:9px;background:#dcd7d2;width:0}
.row.own .bar{background:linear-gradient(90deg,#1d4ed8,#0ea5a0)}
.row .num{position:absolute;right:0;top:0;font-size:56px;font-weight:300;letter-spacing:-.04em;font-variant-numeric:tabular-nums;color:#8c8782;opacity:0}
.row.own .num{color:#0b0b0c}
/* close */
#wave2{position:absolute;left:200px;top:236px;width:880px;height:130px;opacity:0}
#mark i{position:absolute;background:#0b0b0c;opacity:0}
#mark .dotg{background:linear-gradient(135deg,#1d4ed8,#0ea5a0)}
#name{position:absolute;opacity:0}#name svg{height:100%;display:block}
#white{position:absolute;inset:0;background:#fdfcfc;opacity:0}
</style></head><body>
<div class="glow" id="g1"></div><div class="glow" id="g2"></div>

<div class="sc" id="A"><div id="wave"></div><div class="c" id="sent"></div>
  <div class="c cap" id="capA" style="top:604px;opacity:0">Each word gets the probability that a human rater would accept it.</div></div>

<div class="sc" id="B"><div class="c" id="big"></div><div id="phones"></div>
  <div class="c" id="tipl" style="opacity:0">COACHING TIP</div><div class="c" id="tip" style="opacity:0"></div>
  <div class="c cap" id="capB" style="top:92px;opacity:0">Each sound gets one too.</div></div>

<div class="sc" id="F1"><div class="c h" style="top:170px">Ten languages.</div><div id="langs"></div></div>
<div class="sc" id="F2"><div class="c h" style="top:150px">Tones, stress and length.</div><div class="c" id="zh"></div><div class="c cap" id="zhn" style="top:486px;opacity:0"></div></div>
<div class="sc" id="F3"><div class="c h" style="top:130px">On your device.</div><div class="c" id="ms"><span>0</span><small>ms</small></div>
  <div class="c cap" style="top:520px">per sentence on a laptop CPU · no cloud · a 67 MB model</div></div>

<div class="sc" id="R1"><div class="c cap" style="top:124px">__N__ public comparisons with Microsoft Azure Pronunciation Assessment</div><div id="dots"></div>
  <div class="c" id="legend"></div><div class="c h" id="r1h" style="top:560px;font-size:44px;opacity:0"></div></div>
<div class="sc" id="R2"><div class="c cap" style="top:124px">Agreement with expert raters, English learners</div>
  <div class="row own" style="top:232px"><div class="who">Pronunciada</div><div class="bar"></div><div class="num"></div></div>
  <div class="row" style="top:322px"><div class="who">Azure</div><div class="bar"></div><div class="num"></div></div>
  <div class="row" style="top:412px"><div class="who">ZIPA alone<small>the open model it is built on</small></div><div class="bar"></div><div class="num"></div></div>
  <div class="c h" id="r2h" style="top:560px;font-size:40px;opacity:0"></div></div>

<div class="sc" id="Z"><div id="wave2"></div><div id="mark"></div><div id="name"></div>
  <div class="c" id="zh1" style="top:402px;font-size:32px;font-weight:300;letter-spacing:-.02em;opacity:0">On-device pronunciation assessment with cloud-level accuracy.</div>
  <div class="c mono soft" id="zh2" style="top:468px;font-size:19px;opacity:0">Open source · github.com/moglearn/pronunciada</div></div>
<div id="white"></div>

<script>
const D=__DATA__, VOICE=__VOICE__, anims=[], ticks=[], $=s=>document.querySelector(s), EASE='cubic-bezier(.2,.7,.2,1)';
function A(el,kf,t0,d,ease){const a=el.animate(kf,{delay:t0*1000,duration:d*1000,fill:'both',easing:ease||EASE});a.pause();anims.push(a);return a;}
function fade(el,t0,d,rise){A(el,[{opacity:0,transform:`translateY(${rise==null?12:rise}px)`},{opacity:1,transform:'none'}],t0,d||.7);}
function life(el,tin,tout,din,dout){din=din||.8;dout=dout||.6;const T=tout+dout-tin;
  A(el,[{opacity:0,offset:0,easing:EASE},{opacity:1,offset:din/T},{opacity:1,offset:(tout-tin)/T,easing:'ease-in'},{opacity:0,offset:1}],tin,T,'linear');}
function count(el,t0,d,to,dec){ticks.push(t=>{const k=Math.min(1,Math.max(0,(t-t0)/d)),e=1-Math.pow(1-k,3);el.textContent=(to*e).toFixed(dec||0);});}
const mix=(k)=>{const a=[29,78,216],b=[14,165,160];return 'rgb('+a.map((v,i)=>Math.round(v+(b[i]-v)*k)).join(',')+')';};
A($('#g1'),[{transform:'translate(0,0)'},{transform:'translate(-150px,110px)'}],0,__LEN__,'linear');
A($('#g2'),[{transform:'translate(0,0)'},{transform:'translate(170px,-90px)'}],0,__LEN__,'linear');

/* A: the recording, its words, their scores */
life($('#A'),0,7.7,.6,.7);
const wave=$('#wave'), n=D.levels.length, lowA=D.low.start, lowB=D.low.end;
D.levels.forEach((v,i)=>{const h=Math.max(3,v*124), b=document.createElement('i');b.style.cssText=`left:${i*6}px;height:${h}px;top:${65-h/2}px`;wave.appendChild(b);
  const t=VOICE+(D.times[i]-D.a0);
  A(b,[{transform:`scaleY(${Math.min(1,3/h)})`,opacity:.22},{transform:'scaleY(1)',opacity:1}],t,.22);
  if(D.times[i]>=lowA-.02&&D.times[i]<=lowB+.02) A(b,[{backgroundColor:'#0b0b0c'},{backgroundColor:mix((D.times[i]-lowA)/(lowB-lowA))}],5.0,.8);});
const sent=$('#sent');
D.words.forEach((w,i)=>{const low=w.prob<.5, e=document.createElement('span');e.className='wd';
  e.innerHTML=`<span class="t"><span class="k">${w.text}</span>${low?`<span class="g grad">${w.text}</span>`:''}</span><span class="p">${w.prob.toFixed(2)}</span>`;sent.appendChild(e);
  fade(e,VOICE+(w.start-D.a0)-.05,.45,10);
  fade(e.querySelector('.p'),3.7+i*.09,.6,8);
  if(low){A(e.querySelector('.k'),[{opacity:1},{opacity:0}],5.0,.8);A(e.querySelector('.g'),[{opacity:0},{opacity:1}],5.0,.8);
          A(e.querySelector('.p'),[{color:'#8c8782'},{color:'#1d4ed8'}],5.0,.8);}});
life($('#capA'),4.0,7.2,.8,.5);

/* B: the flagged word, sound by sound */
life($('#B'),8.2,16.0,.7,.7);
const big=$('#big');big.innerHTML=`<span class="grad">${D.low.text}</span>`;
A(big,[{opacity:1,transform:'scale(1)',letterSpacing:'-.045em',offset:0},{opacity:1,transform:'scale(1)',letterSpacing:'-.045em',offset:.45},{opacity:0,transform:'scale(1.02)',letterSpacing:'.12em',offset:1}],8.2,2.0);
const P=D.low.phones, ph=$('#phones');
P.forEach((p,i)=>{const e=document.createElement('div');e.className='ph';e.style.left=(640-95+(i-(P.length-1)/2)*200)+'px';
  const lowp=p.prob<.5, note=p.heard==null?'':(p.heard===''?'not heard':'heard as '+p.heard);
  e.innerHTML=`<span class="i">${p.ipa}</span><span class="dot"></span><span class="n" style="color:${lowp?'#1d4ed8':'#0b0b0c'}">${p.prob.toFixed(2)}</span><span class="x">${note}</span>`;ph.appendChild(e);
  A(e,[{opacity:0,transform:`translateX(${-(i-(P.length-1)/2)*70}px)`},{opacity:1,transform:'none'}],9.3,1.0);
  fade(e.querySelector('.n'),10.3+i*.14,.6,8);
  if(note) fade(e.querySelector('.x'),11.9+i*.1,.6,6);
  if(p.heard===''){A(e.querySelector('.i'),[{opacity:1,transform:'scale(1)'},{opacity:0,transform:'scale(.2)'}],11.1,.7,'ease-in');
                   A(e.querySelector('.dot'),[{opacity:0,transform:'scale(.2)'},{opacity:1,transform:'scale(1)'}],11.5,.7);}});
life($('#capB'),9.6,15.4,.8,.5);
fade($('#tipl'),12.9,.6,6);$('#tip').textContent=D.tip;
A($('#tip'),[{opacity:0,clipPath:'inset(0 100% 0 0)'},{opacity:1,clipPath:'inset(0 0% 0 0)'}],13.1,1.5,'cubic-bezier(.4,0,.2,1)');

/* F: three features */
life($('#F1'),16.6,20.7);life($('#F2'),21.3,25.5);life($('#F3'),26.1,30.0);
['English','中文','日本語','Español','Français','Deutsch','Italiano','Português','Русский','한국어'].forEach((t,i)=>{const e=document.createElement('span');
  e.innerHTML=t+'<b>'+t+'</b>';$('#langs').appendChild(e);A(e.querySelector('b'),[{opacity:0},{opacity:1}],17.3+i*.26,.5);});
D.zh.forEach((s,i)=>{const e=document.createElement('span');
  e.innerHTML=s.text+(s.low?`<span class="g grad">${s.text}</span>`:'');$('#zh').appendChild(e);
  if(s.low){A(e.querySelector('.g'),[{opacity:0},{opacity:1}],22.5,.8);}});
$('#zhn').innerHTML=D.zh.filter(s=>s.low).map(s=>`${s.text}  tone ${s.exp} expected, tone ${s.heard} heard`).join('   ·   ');fade($('#zhn'),23.2,.8,8);
count($('#ms span'),26.5,1.3,50,0);

/* R: results */
life($('#R1'),30.6,37.3);life($('#R2'),37.9,44.2);
const T=D.tally, dots=$('#dots');
for(let i=0;i<T.n;i++){const e=document.createElement('i'), col=i%14,row=Math.floor(i/14);e.style.cssText=`left:${col*41}px;top:${row*42.4}px`;dots.appendChild(e);
  A(e,[{transform:'scale(0)'},{transform:'scale(1)'}],31.0+(col+row)*.045,.5);
  if(i<T.better) A(e,[{backgroundColor:'#e9e5e1'},{backgroundColor:mix(col/13)}],32.4+i*.028,.5);
  else if(i>=T.n-T.behind) A(e,[{backgroundColor:'#e9e5e1'},{backgroundColor:'#8c8782'}],34.7+(i-(T.n-T.behind))*.05,.5);}
$('#legend').innerHTML=`<span><b>${T.better}</b> significantly better</span><span><b>${T.same}</b> no significant difference</span><span><b>${T.behind}</b> behind</span>`;
[...$('#legend').children].forEach((e,i)=>fade(e,[32.6,33.8,34.8][i],.6,6));
$('#r1h').innerHTML=`Level with or ahead in <span class="grad">${T.better+T.same} of ${T.n}</span>.`;fade($('#r1h'),35.5,.8);
[['ours',38.4],['azure',39.2],['zipa',40.4]].forEach(([k,t],i)=>{const r=document.querySelectorAll('.row')[i], v=D.bars[k];
  A(r,[{opacity:0},{opacity:1}],t-.3,.6);A(r.querySelector('.bar'),[{width:'0px'},{width:(v*640)+'px'}],t,1.2);
  A(r.querySelector('.num'),[{opacity:0},{opacity:1}],t+.1,.6);count(r.querySelector('.num'),t,1.2,v,2);});
$('#r2h').innerHTML=`Significantly better than ZIPA alone on <span class="grad">${D.zipa.ahead} of ${D.zipa.n}</span> tests.`;fade($('#r2h'),41.6,.8);

/* Z: the waveform becomes the logo */
life($('#Z'),44.6,__LEN__,.5,.1);
const w2=$('#wave2');D.levels.forEach((v,i)=>{const h=Math.max(3,v*124),b=document.createElement('i');b.style.cssText=`left:${i*6}px;height:${h}px;top:${65-h/2}px`;w2.appendChild(b);});
A(w2,[{opacity:0,offset:0},{opacity:1,offset:.35},{opacity:1,offset:.6},{opacity:0,offset:1}],44.6,1.9,'linear');
const H=81.6, MH=H*.8/1.02, bw=MH*.105/.8, gap=MH*.075/.8, mw=5*bw+4*gap, vb=D.name.match(/viewBox="0 -?[\d.]+ ([\d.]+) ([\d.]+)"/), nw=H*vb[1]/vb[2], lock=mw+H*.3/1.02+nw, X0=640-lock/2, Y0=262, CY=Y0+H*.412;
const name=$('#name');name.innerHTML=D.name;name.style.cssText=`left:${X0+mw+H*.3/1.02}px;top:${Y0}px;height:${H}px`;
D.shape.forEach((s,i)=>{const pos=[.14,.31,.5,.69,.86][i], idx=Math.round(pos*(n-1)), h0=Math.max(8,D.levels[idx]*124), x0=200+idx*6, dot=s==null;
  const fh=dot?bw*1.32:s*MH, fw=dot?bw*1.32:bw, fx=X0+i*(bw+gap)+(dot?(bw-fw)/2:0);
  const mk=(cls)=>{const e=document.createElement('i');if(cls)e.className=cls;$('#mark').appendChild(e);return e;};
  const kf=[{left:x0+'px',top:(301-h0/2)+'px',width:'3px',height:h0+'px',borderRadius:'2px'},{left:fx+'px',top:(CY-fh/2)+'px',width:fw+'px',height:fh+'px',borderRadius:(fw/2)+'px'}];
  const e=mk();A(e,[{opacity:0},{opacity:1}],45.5,.5);A(e,kf,46.3,1.3,'cubic-bezier(.6,0,.2,1)');
  if(dot){const g=mk('dotg');A(g,kf,46.3,1.3,'cubic-bezier(.6,0,.2,1)');A(g,[{opacity:0},{opacity:1}],47.4,.6);A(e,[{opacity:1},{opacity:0}],47.5,.6).effect.updateTiming({fill:'forwards'});}});
A(name,[{opacity:0,transform:'translateX(-14px)'},{opacity:1,transform:'none'}],47.5,.9);
fade($('#zh1'),48.4,.9,10);fade($('#zh2'),49.2,.9,8);
A($('#white'),[{opacity:0},{opacity:1}],50.7,1.1,'ease-in-out');
window.seek=t=>{anims.forEach(a=>a.currentTime=t*1000);ticks.forEach(f=>f(t));};seek(0);
</script></body></html>"""


def page_html() -> str:
    d = data()
    return (PAGE.replace("__FONTS__", font_css()).replace("__DATA__", json.dumps(d, ensure_ascii=False)).replace("__VOICE__", str(VOICE_AT))
            .replace("__LEN__", str(LENGTH)).replace("__N__", str(d["tally"]["n"])))


def sound(path: Path) -> None:
    """The recording over a quiet bed: slow sine chords with a gentle swell, generated here."""
    sr = 44100
    t = np.arange(int(LENGTH * sr)) / sr
    bed = np.zeros_like(t)
    chords = [(146.83, 220.0, 329.63, 369.99), (123.47, 220.0, 293.66, 369.99), (98.0, 196.0, 293.66, 369.99), (110.0, 220.0, 329.63, 440.0)]
    span = LENGTH / len(chords)
    for k, notes in enumerate(chords):
        env = np.clip(1 - np.abs((t - (k + 0.5) * span) / (span * 0.75)), 0, 1) ** 1.5       # overlapping swells
        for j, f in enumerate(notes):
            for detune in (-0.6, 0.6):
                bed += env * np.sin(2 * np.pi * (f + detune) * t + j) * (0.5 if j else 0.8) / (1 + j * 0.35)
    bed *= 1 + 0.12 * np.sin(2 * np.pi * 0.11 * t)
    bed *= np.clip(t / 3.0, 0, 1) * np.clip((LENGTH - t) / 3.5, 0, 1)
    bed = 0.07 * bed / np.abs(bed).max()
    d = data()
    voice, vsr = sf.read(str(SAMPLE.with_suffix(".mp3")), dtype="float32", always_2d=True)
    v = voice.mean(axis=1)[int(d["a0"] * vsr):int((d["a0"] + d["dur"]) * vsr)]
    v = np.interp(np.arange(int(len(v) * sr / vsr)) * vsr / sr, np.arange(len(v)), v)
    ramp = int(0.04 * sr)
    v[:ramp] *= np.linspace(0, 1, ramp); v[-ramp * 4:] *= np.linspace(1, 0, ramp * 4)
    at = int(VOICE_AT * sr)
    duck = np.ones_like(t); duck[at - sr // 2:at + len(v) + sr // 2] = 0.45
    duck = np.convolve(duck, np.ones(sr // 3) / (sr // 3), mode="same")
    mix = bed * duck
    mix[at:at + len(v)] += 0.9 * v / np.abs(v).max() * 0.7
    sf.write(str(path), np.stack([mix, mix], axis=1), sr)


def main(sheet: str | None) -> None:
    with tempfile.TemporaryDirectory() as d, sync_playwright() as play:
        tmp = Path(d)
        (tmp / "film.html").write_text(page_html(), encoding="utf-8")
        browser = play.chromium.launch(channel="chrome", headless=True)
        page = browser.new_context(viewport={"width": 1280, "height": 720}, device_scale_factor=1.5).new_page()
        page.goto((tmp / "film.html").as_uri())
        page.wait_for_function("window.seek && document.fonts.ready.then(() => true)")
        if sheet:
            times = [1.6, 2.6, 4.6, 6.4, 9.0, 10.2, 12.4, 15.0, 19.0, 24.4, 28.0, 32.0, 36.4, 41.0, 43.2, 45.9, 46.9, 49.8]
            for i, t in enumerate(times):
                page.evaluate(f"seek({t})")
                page.screenshot(path=str(tmp / f"s{i:02d}.png"))
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp / "s%02d.png"), "-vf", "scale=640:-1,tile=3x6", sheet], check=True)
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
    print(f"wrote {OUT.relative_to(ROOT)} ({LENGTH:.0f} s, {OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--sheet" else None)
