"""Loopback HTTP/SSE bridge to the resident native NPU capture engine."""
from __future__ import annotations
import argparse
import base64
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit, parse_qs
import uuid
import queue
import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/open_qwen36'))
from capture import prepare_cases


class Engine:
    def __init__(self, args):
        self.args, self.lock = args, threading.Lock()
        self.state, self.current, self.proc = 'loading', None, None
        self.runs = {}
        self.jobs = queue.Queue()
        self.manifest = json.loads((args.kernels/'manifest.json').read_text(encoding='utf-8'))
        self.restore()
        threading.Thread(target=self.worker, daemon=True).start()
        self.tokenizer = Tokenizer.from_file(str(args.model / 'tokenizer.json'))
        threading.Thread(target=self.boot, daemon=True).start()

    def boot(self):
        try:
            env = os.environ.copy()
            if os.name == 'nt':
                env['PATH'] = str(Path(os.environ.get('SystemRoot', 'C:/Windows')) / 'System32/AMD') + os.pathsep + env.get('PATH', '')
            self.proc = subprocess.Popen([str(self.args.exe.resolve()), '--model', str(self.args.model.resolve()),
                '--kernels', str(self.args.kernels.resolve()), '--max-ctx', str(self.args.max_ctx), '--capture-server', '--pmode', 'none'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding='utf-8', env=env, bufsize=1)
            reply = json.loads(self.proc.stdout.readline())
            if not reply.get('ready'):
                raise RuntimeError('engine did not become ready')
            with self.lock:
                self.state = 'ready'
        except Exception:
            with self.lock:
                self.state = 'error'

    def capabilities(self):
        spec = self.manifest['spec']
        return {'schema_version':1, 'protocol':'openflowlm.capture.v1', 'request_schema':'/api/v1/schema', 'backend':'native_openflowlm',
                'model':{'family':self.manifest['family'], 'layers':spec['num_layers'],
                         'hidden':spec['hidden'], 'experts':256, 'shared_experts':1, 'topk':8, 'vocab':spec['real_vocab'],
                         'layer_types':spec['layer_types']},
                'context_capacity':self.args.max_ctx, 'features':self.manifest.get('moe_capture',{}).get('capabilities',[]),
                'capture':{'routes':'all input positions and layers', 'vector_layers':'selectable',
                           'shared_gate_frequency':'every executed position and layer, independent of vector_layers',
                           'capture_positions':'prompt indices or all', 'micro_layers':'selectable, multiple layers or all selected vector layers',
                           'micro_rank':[0,7], 'ablation':'one position and layer per case, baseline TopK ranks'},
                'execution':{'queue':'FIFO', 'concurrency':1, 'cancel':'next token boundary',
                             'case_limit':None, 'generation_limit':'context_capacity minus prompt length'},
                'transport':{'events':'SSE', 'reconnect':'Last-Event-ID or after query',
                             'compact_route':'routing.topk with native selected logits, probabilities and dispatched weights',
                             'progress':'named progress heartbeat, no SSE id, approximately once per second',
                             'tensor_modes':['inline','references'], 'dtypes':['f32','i32','bf16'],
                             'byte_order':'little', 'request_bytes':self.args.max_request_bytes}}

    def request_schema(self):
        integer = lambda upper: {'type':'integer','minimum':0,'maximum':upper}
        indices = lambda upper: {'type':'array','uniqueItems':True,'items':integer(upper)}
        selection = lambda upper: {'oneOf':[indices(upper),{'const':'all'}]}
        return {'$schema':'https://json-schema.org/draft/2020-12/schema','title':'OpenFlowLM capture request v1',
                'type':'object','additionalProperties':False,'required':['cases'],
                'properties':{'compare':{'type':'boolean','default':False},
                    'max_tokens':{'type':'integer','minimum':1,'maximum':self.args.max_ctx-1},
                    'cases':{'type':'array','minItems':1,'items':{'type':'object','additionalProperties':False,
                        'required':['text'],'properties':{'id':{'type':'string','minLength':1},'text':{'type':'string','minLength':1},
                        'focus':{'type':'string','minLength':1},'continuation':{'type':'string','minLength':1},
                        'capture_positions':selection(self.args.max_ctx-1),'vector_layers':selection(39),
                        'micro_layers':selection(39),'micro_rank':integer(7),'target_position':integer(self.args.max_ctx-1),
                        'ablate_layer':integer(39),'ablation_ranks':indices(7),
                        'max_tokens':{'type':'integer','minimum':1,'maximum':self.args.max_ctx-1}}}}}}

    def tokenize(self, payload):
        if not isinstance(payload,dict) or set(payload)-{'text','focus'} or not isinstance(payload.get('text'),str):
            raise ValueError('expected text and optional focus')
        c=prepare_cases([dict(payload,id='tokenize')],self.tokenizer,1)['cases'][0]
        return {'schema_version':1,'input_ids':c['ids'],'tokens':[
            dict(t,position=i,display=self.tokenizer.decode([t['id']],skip_special_tokens=False)) for i,t in enumerate(c['tokens'])],
            'capture_positions':c['capture_positions'],'target_position':c['target_position']}

    def public_run(self, run_id):
        r = self.runs[run_id]
        return dict(schema_version=1,run_id=run_id, **{k:r.get(k) for k in ('state','created_at','started_at','finished_at','request')})

    def save(self, run_id):
        r = self.runs[run_id]
        temporary = r['directory']/'state.tmp'
        temporary.write_text(json.dumps(self.public_run(run_id),ensure_ascii=False),encoding='utf-8')
        temporary.replace(r['directory']/'state.json')

    def restore(self):
        self.args.output.mkdir(parents=True,exist_ok=True)
        for directory in self.args.output.iterdir():
            if not directory.is_dir() or len(directory.name)!=32 or any(c not in '0123456789abcdef' for c in directory.name):
                continue
            state_file = directory/'state.json'
            if state_file.exists():
                r = json.loads(state_file.read_text(encoding='utf-8'))
            else:
                r = {'state':'incomplete', 'created_at':directory.stat().st_mtime, 'request':None}
            events = directory/'events.jsonl'
            complete = False
            if events.exists():
                with events.open('rb') as f:
                    f.seek(max(0,events.stat().st_size-4096))
                    lines=f.read().splitlines()
                if lines:
                    try: complete=json.loads(lines[-1]).get('kind')=='complete'
                    except ValueError: pass
            if complete: r['state']='complete'
            elif r['state'] in ('running','queued','cancelling','complete'): r['state']='incomplete'
            self.runs[directory.name]=dict(r,directory=directory)
            self.save(directory.name)
        if self.runs: self.current=max(self.runs,key=lambda k:self.runs[k]['created_at'])

    def start(self, payload):
        if set(payload)-{'cases','max_tokens','compare'}:
            raise ValueError('unsupported request field')
        cases = payload.get('cases')
        if not isinstance(cases,list) or not cases: raise ValueError('cases must be nonempty')
        maximum = payload.get('max_tokens',8)
        if type(maximum) is not int or not 1<=maximum<self.args.max_ctx: raise ValueError('generation exceeds context capacity')
        if type(payload.get('compare',False)) is not bool: raise ValueError('compare must be boolean')
        allowed={'id','text','focus','continuation','capture_positions','target_position','vector_layers','micro_layers','micro_rank','ablate_layer','ablation_ranks','max_tokens'}
        compiled=[]
        for i,c in enumerate(cases):
            if not isinstance(c,dict) or set(c)-allowed or not isinstance(c.get('text'),str): raise ValueError('invalid case')
            compiled.append(dict(c,id=c.get('id',f'live_{i}')))
        config=prepare_cases(compiled,self.tokenizer,maximum)
        for c in config['cases']:
            if len(c['ids'])+max(c['max_tokens'],len(c.get('continuation_ids',[])))>self.args.max_ctx:
                raise ValueError('prompt and continuation exceed context capacity')
        config.update(live=True,compare=payload.get('compare',False))
        with self.lock:
            if self.state=='error': raise RuntimeError('native engine unavailable')
            run_id=uuid.uuid4().hex
            directory=self.args.output/run_id; directory.mkdir()
            (directory/'config.json').write_text(json.dumps(config,ensure_ascii=False),encoding='utf-8')
            self.runs[run_id]={'directory':directory,'state':'queued','created_at':time.time(),'request':payload}
            self.current=run_id; self.save(run_id); self.jobs.put(run_id)
            return run_id

    def cancel(self, run_id):
        with self.lock:
            run=self.runs[run_id]
            if run['state']=='queued':
                run['state']='cancelled'; run['finished_at']=time.time()
            elif run['state'] in ('running','cancelling'):
                (run['directory']/'cancel').touch(); run['state']='cancelling'
            self.save(run_id)
            return self.public_run(run_id)

    def worker(self):
        while True:
            run_id=self.jobs.get(); run=self.runs[run_id]
            while self.state=='loading': time.sleep(0.1)
            with self.lock:
                if run['state']=='cancelled': continue
                if self.state=='error':
                    run['state']='error'; run['finished_at']=time.time(); self.save(run_id); continue
                run['state']='running'; run['started_at']=time.time(); self.state='running'; self.save(run_id)
            try:
                self.proc.stdin.write(json.dumps({'config':str((run['directory']/'config.json').resolve()),'output':str(run['directory'].resolve())})+'\n')
                self.proc.stdin.flush(); reply=json.loads(self.proc.stdout.readline())
                with self.lock:
                    run['state']='complete' if reply.get('ok') else 'cancelled' if run['state']=='cancelling' else 'error'
            except Exception:
                with self.lock: run['state']='error'
            finally:
                with self.lock:
                    run['finished_at']=time.time(); self.save(run_id)
                    self.state='ready' if self.proc and self.proc.poll() is None else 'error'

    def status(self):
        with self.lock:
            if self.proc and self.proc.poll() is not None:
                self.state = 'error'
            return {'schema_version':1, 'state':self.state, 'run_id':self.current, 'run_state':self.runs[self.current]['state'] if self.current else None,
                    'backend':'native_openflowlm'}

    def packet(self, event, directory, run_id, inline=True):
        packet = dict(event,schema_version=1,protocol='openflowlm.capture.v1',run_id=run_id)
        if event['kind'] == 'metadata':
            packet['shared_experts'] = 1
        if event['kind'] in ('route','logits'):
            packet['display'] = self.tokenizer.decode([event['input_token_id']],skip_special_tokens=False)
        if 'next_token_id' in event:
            packet['next_token_display'] = self.tokenizer.decode([event['next_token_id']],skip_special_tokens=False)
        if event['kind'] == 'case':
            packet['tokens'] = [dict(t,display=self.tokenizer.decode([t['id']],skip_special_tokens=False),position=i) for i,t in enumerate(event['tokens'])]
        if event.get('tensors'):
            with (directory/'tensors.bin').open('rb') as stream:
                if event['kind'] == 'logits':
                    t = event['tensors'][0]
                    stream.seek(t['offset']); raw = stream.read(t['length'])
                    if len(raw) != t['length']:
                        raise RuntimeError('published tensor is incomplete')
                    p = np.frombuffer(raw, dtype='<f4').astype(np.float64)
                    p = np.exp(p-p.max()); p /= p.sum()
                    best = np.argsort(p)[-16:][::-1]
                    packet['tensors'] = [dict(t,href=f'/api/v1/runs/{run_id}/tensors/{event["event_seq"]}/0')]
                    packet['prediction'] = [{'id':int(i), 'text':self.tokenizer.decode([int(i)],skip_special_tokens=False),'probability':float(p[i])} for i in best]
                    packet['prediction_source'] = 'derived'
                    packet['generated_text'] = self.tokenizer.decode([event['next_token_id']],skip_special_tokens=False)
                else:
                    packet['tensors'] = []
                    arrays = {}
                    for index,t in enumerate(event['tensors']):
                        stream.seek(t['offset']); raw = stream.read(t['length'])
                        if len(raw) != t['length']:
                            raise RuntimeError('published tensor is incomplete')
                        if event['kind'] in ('route','moe_output'):
                            arrays[(t['name'],t.get('slot'))] = np.frombuffer(raw,dtype='<i4' if t['dtype']=='i32' else '<f4')
                        packet['tensors'].append(dict(t,href=f'/api/v1/runs/{run_id}/tensors/{event["event_seq"]}/{index}', **({'data':base64.b64encode(raw).decode('ascii')} if inline else {})))
                    if event['kind']=='route':
                        ids=arrays[('expert_ids',None)]; logits=arrays[('router_logits',None)]
                        probabilities=arrays[('router_probabilities',None)]; weights=arrays[('expert_weights',None)]
                        packet['routing']={'source':'derived','value_source':'native_openflowlm',
                            'topk':[{'id':int(i),'logit':float(logits[i]),'prob':float(probabilities[i]),'weight':float(w)} for i,w in zip(ids,weights)]}
                    if event['kind']=='moe_output':
                        norm=lambda v:float(np.linalg.norm(v.astype(np.float64)))
                        shared=arrays[('shared_output',8)]
                        gate=float(arrays[('shared_gate',None)][0])
                        packet['norms']={'source':'derived','routed_mixture_l2':norm(arrays[('routed_mixture',None)]),
                            'shared_output_l2':norm(shared),'gated_shared_output_l2':abs(gate)*norm(shared),
                            'residual_after_l2':norm(arrays[('residual_after',None)]),
                            'expert_output_l2':[norm(arrays[('expert_output',i)]) for i in range(8)]}
        return packet


class Handler(SimpleHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def __init__(self, request, client_address, server, **kwargs):
        super().__init__(request, client_address, server, directory=str(server.engine.args.web), **kwargs)

    def log_message(self, *args):
        pass

    def local_request(self):
        port=self.server.server_port
        origin=self.headers.get('Origin')
        return self.headers.get('Host') in {f'127.0.0.1:{port}',f'localhost:{port}'} and origin in {None,f'http://127.0.0.1:{port}',f'http://localhost:{port}',*self.server.engine.args.allow_origin}

    def end_headers(self):
        origin=self.headers.get('Origin')
        if origin and self.local_request():
            self.send_header('Access-Control-Allow-Origin',origin)
            self.send_header('Vary','Origin')
        super().end_headers()

    def do_OPTIONS(self):
        if not self.local_request(): return self.respond({'error':'origin not allowed'},403)
        self.send_response(204)
        self.send_header('Access-Control-Allow-Methods','GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers','Content-Type, Last-Event-ID')
        self.send_header('Content-Length','0'); self.end_headers()

    def respond(self, value, code=200):
        data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
        self.send_response(code); self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Cache-Control','no-store'); self.send_header('Content-Length',str(len(data)))
        self.end_headers(); self.wfile.write(data)

    def do_POST(self):
        if not self.local_request():
            return self.respond({'error':'请求来源或接口不受支持'},403)
        try:
            length = int(self.headers.get('Content-Length','0'))
            if not 0 <= length <= self.server.engine.args.max_request_bytes:
                self.close_connection = True
                raise ValueError('request body exceeds configured capacity')
            body = self.rfile.read(length)
            parts=urlsplit(self.path).path.strip('/').split('/')
            if len(parts)==5 and parts[:3]==['api','v1','runs'] and parts[4]=='cancel':
                return self.respond(self.server.engine.cancel(parts[3]),202)
            if self.path not in ('/api/v1/runs','/api/v1/tokenize'): return self.respond({'error':'接口不存在'},404)
            if not 0 < length <= self.server.engine.args.max_request_bytes or not self.headers.get('Content-Type','').startswith('application/json'):
                raise ValueError('需要有效 JSON 请求')
            payload = json.loads(body)
            if self.path=='/api/v1/tokenize': return self.respond(self.server.engine.tokenize(payload))
            run_id = self.server.engine.start(payload)
            self.respond({'schema_version':1,'run_id':run_id},202)
        except KeyError:
            self.respond({'error':'运行或必需字段不存在'},404)
        except (ValueError,TypeError,AttributeError):
            self.respond({'error':'输入配置无效，请查询 capabilities 与协议文档'},400)
        except RuntimeError as e:
            self.respond({'error':str(e)},409)

    def do_GET(self):
        if not self.local_request():
            return self.respond({'error':'仅允许本地同源访问'},403)
        path = urlsplit(self.path).path
        engine = self.server.engine
        if path == '/api/v1/capabilities': return self.respond(engine.capabilities())
        if path == '/api/v1/schema': return self.respond(engine.request_schema())
        if path == '/api/v1/runs':
            with engine.lock: runs=[engine.public_run(k) for k in sorted(engine.runs,key=lambda k:engine.runs[k]['created_at'],reverse=True)]
            query=parse_qs(urlsplit(self.path).query)
            try: offset=max(0,int(query.get('offset',['0'])[0])); limit=max(1,min(1000,int(query.get('limit',['100'])[0])))
            except ValueError: return self.respond({'error':'invalid pagination'},400)
            return self.respond({'schema_version':1,'runs':runs[offset:offset+limit],'total':len(runs),'next_offset':offset+limit if offset+limit<len(runs) else None})
        if path == '/api/v1/status':
            return self.respond(engine.status())
        parts = path.strip('/').split('/')
        if len(parts)>=4 and parts[:3]==['api','v1','runs']:
            run_id=parts[3]
            run = engine.runs.get(run_id)
            if not run:
                return self.respond({'error':'运行不存在'},404)
            if len(parts)==4: return self.respond(engine.public_run(run_id))
            if len(parts)==7 and parts[4]=='tensors':
                try:
                    seq,index=int(parts[5]),int(parts[6])
                    if seq<0 or index<0: raise ValueError()
                    with (run['directory']/'events.jsonl').open(encoding='utf-8') as events:
                        event=next(json.loads(line) for line in events if line.endswith('\n') and json.loads(line)['event_seq']==seq)
                    t=event['tensors'][index]
                    with (run['directory']/'tensors.bin').open('rb') as f:
                        f.seek(t['offset']); raw=f.read(t['length'])
                    if len(raw)!=t['length']: raise ValueError()
                except (ValueError,StopIteration,IndexError,KeyError,FileNotFoundError): return self.respond({'error':'tensor not published'},404)
                self.send_response(200); self.send_header('Content-Type','application/octet-stream')
                self.send_header('Content-Length',str(len(raw))); self.end_headers(); self.wfile.write(raw); return
            if len(parts)!=5 or parts[4]!='events': return self.respond({'error':'接口不存在'},404)
            query=parse_qs(urlsplit(self.path).query)
            mode=query.get('tensors',['inline'])[0]
            if mode not in ('inline','references'): return self.respond({'error':'invalid tensor mode'},400)
            after_value=self.headers.get('Last-Event-ID',query.get('after',['-1'])[0])
            try: after=int(after_value)
            except ValueError: return self.respond({'error':'invalid cursor'},400)
            self.send_response(200); self.send_header('Content-Type','text/event-stream; charset=utf-8')
            self.send_header('Cache-Control','no-store'); self.send_header('Connection','close'); self.end_headers()
            stream = None
            last_progress = 0.0
            cursor = after
            context = {}
            try:
                while True:
                    now = time.monotonic()
                    if now-last_progress >= 1:
                        progress={'schema_version':1,'protocol':'openflowlm.capture.v1','run_id':run_id,
                                  'state':run['state'],'last_event_seq':cursor,'context':context}
                        self.wfile.write(('event: progress\ndata: '+json.dumps(progress,ensure_ascii=False)+'\n\n').encode('utf-8'))
                        self.wfile.flush(); last_progress=now
                    if stream is None and (run['directory']/'events.jsonl').exists():
                        stream = (run['directory']/'events.jsonl').open('r',encoding='utf-8')
                    mark = stream.tell() if stream else 0
                    line = stream.readline() if stream else ''
                    if line.endswith('\n'):
                        event = json.loads(line)
                        cursor = max(cursor,event['event_seq'])
                        if event['kind'] in ('case','run'): context = {}
                        for field in ('case','run','phase','position','input_token_id','layer'):
                            if field in event and not (field=='layer' and event[field]<0): context[field]=event[field]
                        if event['event_seq'] > after:
                            packet = engine.packet(event,run['directory'],run_id,mode=='inline')
                            packet['host_elapsed_ms'] = round((time.time()-run.get('started_at',run['created_at']))*1000)
                            self.wfile.write(f"id: {event['event_seq']}\ndata: {json.dumps(packet,ensure_ascii=False,allow_nan=False)}\n\n".encode('utf-8')); self.wfile.flush()
                        if event['kind']=='complete':
                            break
                    else:
                        if stream: stream.seek(mark)
                        if run['state'] in ('error','cancelled','incomplete','complete'):
                            self.wfile.write(('event: failure\ndata: '+json.dumps({'schema_version':1,'run_id':run_id,'state':run['state'],'error':'capture has no normal completion record'})+'\n\n').encode('utf-8')); self.wfile.flush(); break
                        self.wfile.write(b': waiting\n\n'); self.wfile.flush()
                        time.sleep(0.1)
            except (BrokenPipeError,ConnectionResetError):
                pass
            except Exception:
                try:
                    self.wfile.write(b'event: failure\ndata: {"error":"Live data could not be read"}\n\n'); self.wfile.flush()
                except OSError:
                    pass
            finally:
                if stream: stream.close()
                self.close_connection = True
            return
        if path.startswith('/api/'):
            return self.respond({'error':'接口不存在'},404)
        # Serve only built web artifacts, never the repository or capture directory.
        super().do_GET()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--port',type=int,default=4180)
    ap.add_argument('--max-ctx',type=int,default=None)
    ap.add_argument('--allow-origin',action='append',default=[])
    ap.add_argument('--max-request-bytes',type=int,default=16*1024*1024)
    ap.add_argument('--model',type=Path,default=ROOT/'models/Qwen3.6-35B-A3B-NPU2')
    ap.add_argument('--kernels',type=Path,default=ROOT/'src/xclbins/Qwen3.6-35B-A3B-NPU2/open_kernels_capture')
    ap.add_argument('--exe',type=Path,default=ROOT/'src/open_qwen36/out'/('open_qwen36_cli.exe' if os.name=='nt' else 'open_qwen36_cli'))
    ap.add_argument('--web',type=Path,default=ROOT/'web/moe-lesson/dist')
    ap.add_argument('--output',type=Path,default=ROOT/'captures/live')
    args = ap.parse_args()
    manifest=json.loads((args.kernels/'manifest.json').read_text(encoding='utf-8'))
    args.max_ctx=args.max_ctx or manifest.get('max_ctx_default',4096)
    if args.max_ctx<2 or args.max_request_bytes<1: ap.error('invalid resource capacity')
    server = ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    server.daemon_threads = True
    server.engine = Engine(args)
    print(f'Live MoE: http://127.0.0.1:{args.port}/ (loading NPU model)',flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        proc = server.engine.proc
        if proc and proc.poll() is None:
            proc.terminate(); proc.wait(timeout=10)


if __name__=='__main__':
    main()
