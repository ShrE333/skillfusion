"""Push-to-talk page. The container has no microphone, so the browser records (16 kHz PCM) and POSTs
to /asr. Browsers allow the mic on http://localhost only, so port-forward this port (default 8082)
to your laptop, or serve it behind HTTPS.
"""
from __future__ import annotations

import threading

import numpy as np

PAGE = """<!doctype html><meta charset=utf-8><title>SkillFusion voice</title>
<body style="font-family:system-ui;max-width:32rem;margin:3rem auto;padding:0 1rem">
<h2>SkillFusion voice</h2>
<button id=b style="font-size:1.4rem;padding:1rem 2rem">Hold to talk</button>
<p id=t style="font-size:1.2rem">-</p>
<script>
let ctx, stream, node, chunks=[];
async function start(){ if(!ctx){ stream=await navigator.mediaDevices.getUserMedia({audio:true});
  ctx=new AudioContext({sampleRate:16000}); const src=ctx.createMediaStreamSource(stream);
  node=ctx.createScriptProcessor(4096,1,1); node.onaudioprocess=e=>{ if(rec) chunks.push(new Float32Array(e.inputBuffer.getChannelData(0))) };
  src.connect(node); node.connect(ctx.destination);} chunks=[]; rec=true; b.textContent='Listening...'; }
let rec=false;
async function stop(){ rec=false; b.textContent='Hold to talk'; const n=chunks.reduce((a,c)=>a+c.length,0); const pcm=new Int16Array(n); let o=0;
  for(const c of chunks){ for(let i=0;i<c.length;i++) pcm[o++]=Math.max(-1,Math.min(1,c[i]))*32767; }
  t.textContent='...'; const r=await fetch('/asr',{method:'POST',body:pcm.buffer}); t.textContent=(await r.json()).text; }
b.onmousedown=start; b.onmouseup=stop; b.ontouchstart=start; b.ontouchend=stop;
</script>"""


def make_app(transcribe, on_text):
    """transcribe(np.int16[:]) -> str ; on_text(str) is called with each final transcript."""
    from starlette.applications import Starlette
    from starlette.responses import HTMLResponse, JSONResponse
    from starlette.routing import Route

    async def index(_):
        return HTMLResponse(PAGE)

    async def asr(request):
        pcm = np.frombuffer(await request.body(), dtype="<i2")
        if pcm.size < 3200:                        # < 0.2 s
            return JSONResponse({"text": ""})
        text = transcribe(pcm)
        if text:
            on_text(text)
        return JSONResponse({"text": text})

    return Starlette(routes=[Route("/", index), Route("/asr", asr, methods=["POST"])])


def serve_in_thread(app, port: int = 8082, host: str = "0.0.0.0") -> threading.Thread:
    import uvicorn
    t = threading.Thread(target=uvicorn.run, args=(app,), kwargs=dict(host=host, port=port, log_level="warning"),
                         daemon=True)
    t.start()
    return t
