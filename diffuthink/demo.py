"""Loopback-only demonstration server; intentionally not a production API."""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from .model import generate

PAGE = '''<!doctype html><html lang="fr"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DiffuThink · laboratoire de débruitage</title>
<style>
body{background:#101521;color:#e7eef9;font:17px system-ui;max-width:1000px;margin:60px auto;padding:0 24px}
h1{font-size:56px;margin-bottom:8px}small,.muted{color:#9eafc7}section{background:#1a2435;padding:26px;border-radius:16px;margin:24px 0}
textarea{width:100%;box-sizing:border-box;min-height:100px;background:#101521;color:#e7eef9;border:1px solid #50617b;border-radius:8px;padding:16px;font:20px monospace}
input{width:70px;padding:8px;margin:12px;color:#fff;background:#101521;border:1px solid #50617b}button{background:#9ce7d0;color:#10231d;padding:14px 24px;border:0;border-radius:8px;font-weight:bold;cursor:pointer}
pre{white-space:pre-wrap;overflow-wrap:anywhere}li{padding:8px;font-family:monospace}.error{color:#ffb8b8}
</style><small>MODÈLE ENTRAÎNÉ LOCALEMENT · DIFFUSION DISCRÈTE SIMPLIFIÉE</small>
<h1>DiffuThink</h1><p>Du texte masqué à une phrase reconstruite, étape par étape.</p>
<section><p>Chaque <b>~</b> remplace un octet. Conservez la longueur du mot à retrouver.</p>
<textarea id="template" aria-label="Texte masqué">The c~t runs in the g~rden today.</textarea>
<div><label>Étapes <input id="steps" type="number" min="1" max="64" value="12"></label>
<label>Température <input id="temperature" type="number" min="0" max="2" step="0.1" value="0"></label>
<label>Seed <input id="seed" type="number" value="42"></label></div>
<button id="run">Reconstruire</button><p id="status" role="status"></p><pre id="result"></pre><ol id="trace"></ol></section>
<section><h2>Ce que vous observez</h2><p>Le Transformer regarde les deux côtés des trous. Il prédit les octets manquants et fixe les plus confiants à chaque passage. Le contexte visible reste inchangé.</p>
<p class="muted">Démonstrateur sur une grammaire anglaise synthétique. Pas un assistant conversationnel. Une phrase plausible ne prouve pas une compréhension générale. Température 0 : choix déterministe des octets.</p><pre id="meta"></pre></section>
<script>
const $=id=>document.getElementById(id);
fetch('/api/info').then(r=>r.json()).then(x=>$('meta').textContent=JSON.stringify(x,null,2));
$('run').onclick=async()=>{ $('run').disabled=true; $('status').textContent='Débruitage en cours…'; $('trace').replaceChildren(); $('result').textContent='';
try{const r=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({template:$('template').value,steps:Number($('steps').value),temperature:Number($('temperature').value),seed:Number($('seed').value)})});
const x=await r.json();if(!r.ok)throw Error(x.error);$('result').textContent=x.text;x.trace.forEach(t=>{const li=document.createElement('li');li.textContent=t.text+'  ['+t.remaining+' masqués]';$('trace').append(li)});$('status').textContent='Reconstruction terminée';}
catch(e){$('status').textContent=e.message}finally{$('run').disabled=false}};
</script></html>'''


def serve(model, checkpoint, port):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, body, content_type="application/json; charset=utf-8"):
            raw = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/":
                self.respond(200, PAGE, "text/html; charset=utf-8")
            elif self.path == "/api/info":
                self.respond(200, json.dumps({"parameters": sum(p.numel() for p in model.parameters()), "training_step": checkpoint["step"], "max_bytes": model.config.max_length}))
            else:
                self.respond(404, '{"error":"Not found"}')

        def do_POST(self):
            if self.path != "/api/generate":
                self.respond(404, '{"error":"Not found"}')
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8192:
                    raise ValueError("Request size must be 1..8192 bytes")
                args = json.loads(self.rfile.read(length))
                if not isinstance(args, dict) or not isinstance(args.get("template"), str):
                    raise ValueError("template must be a string")
                steps = int(args.get("steps", 12))
                temperature = float(args.get("temperature", 0))
                if not 1 <= steps <= 64 or not 0 <= temperature <= 2:
                    raise ValueError("steps: 1..64; temperature: 0..2")
                self.respond(200, json.dumps(generate(model, args["template"], steps, temperature, int(args.get("seed", 42)))))
            except (ValueError, TypeError, OverflowError) as exc:
                self.respond(400, json.dumps({"error": str(exc)}))

    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"DiffuThink: http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
