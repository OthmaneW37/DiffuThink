"""Local research demo showing model provenance, sampling and live run metrics."""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
from .inference import load, continue_text, infill
from .train import amp
from .hybrid import generate as generate_causal
from .editor import rewrite_span

PAGE = r'''<!doctype html><html lang="fr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DiffuThink — From Scratch Lab</title>
<style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#0c111a;color:#edf2fb;font:16px system-ui}
main{max-width:1180px;margin:48px auto;padding:0 28px}.eyebrow{color:#84e1c3;letter-spacing:2px;font-size:12px}h1{font-size:clamp(36px,6vw,64px);margin:14px 0}h2{font-size:22px}.muted{color:#97a8c1;line-height:1.7}.stats{display:flex;gap:14px;flex-wrap:wrap}.stat{padding:16px 22px;background:#172131;border:1px solid #2a3b50;border-radius:12px}.stat b{display:block;font-size:25px;color:#fff}
.grid{display:grid;grid-template-columns:1fr 1.15fr;gap:20px;margin-top:24px}.panel{background:#151e2c;border:1px solid #28364b;border-radius:16px;padding:26px}textarea,select,input{font:inherit;background:#0c1421;border:1px solid #42516a;border-radius:8px;color:#edf2fb;padding:12px}textarea{width:100%;min-height:130px;margin:12px 0;resize:vertical}.controls{display:flex;flex-wrap:wrap;gap:14px}.controls label{font-size:12px;color:#aabbce;display:flex;flex-direction:column;gap:6px}.controls input{width:90px}button{border:0;border-radius:8px;background:#96e8cf;color:#10241e;font-weight:700;padding:15px 24px;cursor:pointer;margin-top:20px}button:disabled{opacity:.5}.output{font-size:21px;line-height:1.7;white-space:pre-wrap;overflow-wrap:anywhere;min-height:150px}.trace{font:13px monospace;line-height:1.6;max-height:360px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere}.badge{font-size:12px;color:#b5c4d8}#status{min-height:24px;color:#96e8cf}.hidden{display:none}@media(max-width:800px){.grid{grid-template-columns:1fr}main{padding:0 16px;margin:25px auto}}
</style><main><div class="eyebrow">FROM RANDOM WEIGHTS TO TEXT</div><h1>DiffuThink <span style="color:#96e8cf">Research Lab</span></h1>
<p class="muted">Un modèle hybride entraîné depuis zéro : continuation autorégressive et reconstruction par débruitage. Aucun poids externe préentraîné.</p>
<div class="stats"><div class="stat"><b id="params">…</b>paramètres</div><div class="stat"><b id="stories">…</b>récits d'entraînement</div><div class="stat"><b id="context">…</b>tokens de contexte</div><div class="stat"><b id="step">…</b>mises à jour</div></div>
<div class="grid"><section class="panel"><h2>Expérimenter</h2><label for="mode">Tâche</label> <select id="mode"><option value="continue">Continuer · autorégressif</option><option value="diffusion">Continuer · diffusion expérimentale</option><option value="infill">Compléter entre deux contextes</option></select>
<label for="prompt" style="display:block;margin-top:18px">Début du texte · anglais</label><textarea id="prompt">Once upon a time, a little girl found</textarea>
<div id="suffixbox" class="hidden"><label for="suffix">Suite du texte après le trou</label><textarea id="suffix"> on the balcony.</textarea><label>Tokens à reconstruire <input id="missing" type="number" min="1" max="32" value="2"></label></div>
<div class="controls"><label>Étapes de diffusion<input id="steps" type="number" min="1" max="32" value="12" disabled></label><label>Température<input id="temperature" type="number" min="0" max="1.5" step="0.1" value="0.5"></label><label>Seed<input id="seed" type="number" value="42"></label><label>Budget de tokens<input id="length" type="number" min="1" max="256" value="192"></label><label>Stratégie de diffusion<select id="policy" disabled><option value="confidence">Confiance</option><option value="adaptive">Budget adaptatif</option><option value="left_to_right">Gauche à droite</option></select></label></div>
<p class="badge">Continuation : jusqu'à 32 tokens supplémentaires pour terminer la phrase. Une fin de phrase ne garantit pas une histoire achevée.</p>
<button id="generate">Générer</button><p id="status" role="status"></p><p class="muted">Température 0 : résultat déterministe. 0,5 : variations modérées. La continuation recommandée prédit un token après l'autre. Le modèle reste spécialisé dans les récits anglais simples.</p></section>
<section class="panel"><h2>Texte généré</h2><div id="output" class="output">Le résultat apparaîtra ici.</div><p id="metrics" class="badge"></p><details><summary>Voir les étapes de génération</summary><div id="trace" class="trace"></div></details></section></div>
<section class="panel" style="margin-top:20px"><h2>Deux objectifs, un même réseau</h2><p class="muted">En continuation, l'attention causale interdit de voir les tokens futurs. Pour compléter des trous, l'attention bidirectionnelle utilise les deux côtés. RoPE, RMSNorm et SwiGLU restent communs. La continuation autorégressive n'est pas présentée comme de la diffusion.</p><p class="muted">Données : TinyStories, récits synthétiques générés par GPT-3.5/4. Poids appris dans ce projet. Développement assisté par IA. Des erreurs et répétitions restent possibles ; ce modèle n'est pas un assistant généraliste.</p></section></main>
<script>
const $=x=>document.getElementById(x);
fetch('/api/info').then(r=>r.json()).then(x=>{
 $('params').textContent=(x.parameters/1e6).toFixed(2)+' M';
 $('step').textContent=x.step.toLocaleString();
 $('stories').textContent=x.manifest.splits.train.documents.toLocaleString();
 $('context').textContent=x.manifest.max_length;
});
$('mode').onchange=()=>{
 const causal=$('mode').value==='continue';
 $('suffixbox').classList.toggle('hidden',$('mode').value!=='infill');
 $('steps').disabled=causal;$('policy').disabled=causal;
 $('length').max=causal?256:128;
 if(!causal)$('length').value=Math.min(128,Number($('length').value));
};
$('generate').onclick=async()=>{
 $('generate').disabled=true;$('status').textContent='Génération en cours…';$('trace').textContent='';
 try{
  const r=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
   mode:$('mode').value,prompt:$('prompt').value,suffix:$('suffix').value,missing_tokens:Number($('missing').value),
   max_new_tokens:Number($('length').value),steps:Number($('steps').value),temperature:Number($('temperature').value),
   seed:Number($('seed').value),policy:$('policy').value})});
  const x=await r.json();if(!r.ok)throw Error(x.error);
  $('output').textContent=x.text;
  $('metrics').textContent=(x.mode==='autoregressive'?'Autorégressif':'Débruitage')+' · '+x.forward_passes+' passes · '+Math.round(x.latency_ms)+' ms · seed '+x.seed;
  $('trace').textContent=x.trace.map(t=>(t.mode==='autoregressive'?'Token '+t.step:'Bloc '+(t.block||1)+' / étape '+t.step+' · '+t.remaining+' masqués')+'\n'+t.text).join('\n\n');
  const endings={eos:'Le modèle a émis son signal de fin.',sentence_boundary:'Arrêt en fin de phrase.',length:'Budget atteint : le texte peut rester inachevé.'};
  $('status').textContent=endings[x.stop_reason]||'Génération terminée';
 }catch(e){$('status').textContent=e.message}finally{$('generate').disabled=false}
};
</script></html>'''


def serve(args):
    import torch
    torch.set_num_threads(getattr(args,"threads",4))
    model, tokenizer = load(args.model, args.device)
    ranker = None
    ranker_path = getattr(args, "ranker", None)
    if ranker_path:
        from diffuthink.storypatch.ranker import Reranker
        ranker = Reranker.load(ranker_path, tokenizer, args.device)
    info = json.loads((Path(args.model) / "training_info.json").read_text())
    phase = info.get("previous_phase")
    total_steps = info["step"]
    while phase:
        total_steps += phase["step"]
        phase = phase.get("previous_phase")
    info = {**info, "phase_step": info["step"], "step": total_steps}
    if ranker is not None:
        info["ranker"] = {"parameters": ranker.info["parameters"], "step": ranker.info["step"],
                          "calibration": ranker.info["calibration"]}
    # Warm the kernels before reporting interactive latency to the viewer.
    with torch.inference_mode(), amp(args.device, "bf16" if args.device == "cuda" else "fp32"):
        warm = torch.tensor([[1, 3, 2]], device=args.device)
        model(warm, torch.tensor([0.5], device=args.device))
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, body, mime="application/json; charset=utf-8"):
            raw = body.encode()
            self.send_response(status)
            self.send_header("Content-Type",mime)
            self.send_header("Content-Length",str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        def do_GET(self):
            if self.path == "/": self.respond(200,(Path(__file__).parent/"workshop.html").read_text(encoding="utf-8"),"text/html; charset=utf-8")
            elif self.path == "/lab": self.respond(200,PAGE,"text/html; charset=utf-8")
            elif self.path == "/api/info": self.respond(200,json.dumps(info))
            else: self.respond(404,'{"error":"Not found"}')
        def do_POST(self):
            if self.path != "/api/generate":
                self.respond(404,'{"error":"Not found"}')
                return
            try:
                size=int(self.headers.get("Content-Length",0))
                if not 0 < size <= 16384: raise ValueError("Invalid request size")
                data=json.loads(self.rfile.read(size))
                if not isinstance(data,dict) or not isinstance(data.get("prompt"),str): raise ValueError("prompt must be a string")
                options=dict(steps=int(data.get("steps",12)),temperature=float(data.get("temperature",0.7)),seed=int(data.get("seed",42)),policy=data.get("policy","confidence"),precision="bf16" if args.device=="cuda" else "fp32")
                if not 1<=options["steps"]<=32 or not 0<=options["temperature"]<=1.5: raise ValueError("Invalid sampling range")
                if data.get("mode")=="rewrite":
                    ranking=data.get("ranking","nll")
                    if ranking not in ("nll","learned"):raise ValueError("Unknown ranking mode")
                    if ranking=="learned" and ranker is None:raise ValueError("Experimental ranker is not loaded")
                    result=rewrite_span(model,tokenizer,data["prompt"],data.get("start"),data.get("end"),seed=options["seed"],precision=options["precision"],ranker=ranker if ranking=="learned" else None)
                    result["ranking"]=ranking
                elif data.get("mode")=="infill":
                    if not isinstance(data.get("suffix",""),str): raise ValueError("suffix must be a string")
                    result=infill(model,tokenizer,data["prompt"],data.get("suffix",""),int(data.get("missing_tokens",2)),**options)
                elif data.get("mode") in ("continue", "diffusion"):
                    length=int(data.get("max_new_tokens",64))
                    maximum = 256 if data["mode"] == "continue" else 128
                    if not 1<=length<=maximum: raise ValueError(f"Length must be 1..{maximum}")
                    if data["mode"] == "continue":
                        if info.get("continuation_mode") != "autoregressive":
                            raise ValueError("This checkpoint has no trained causal continuation mode")
                        result=generate_causal(model,tokenizer,data["prompt"],max_new_tokens=length,finish_sentence_tokens=32,**options)
                    else:
                        result=continue_text(model,tokenizer,data["prompt"],max_new_tokens=length,**options)
                else: raise ValueError("Unknown task")
                self.respond(200,json.dumps(result,ensure_ascii=False))
            except (ValueError,TypeError,OverflowError) as exc:
                self.respond(400,json.dumps({"error":str(exc)}))
    server=HTTPServer((getattr(args,"host","127.0.0.1"),args.port),Handler)
    print(f"DiffuThink v2: http://127.0.0.1:{args.port}",flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
