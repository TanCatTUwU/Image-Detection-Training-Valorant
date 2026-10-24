from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

PROJECT=Path(__file__).resolve().parent
FRAME_ID=re.compile(r'frame-[0-9]{6,12}\Z')


class AssistantClient:
    def __init__(self,api_base,token):
        parsed=urlsplit(api_base)
        if parsed.scheme!='http' or parsed.hostname!='127.0.0.1' or not parsed.port or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path.rstrip('/')!='/api/assistant':
            raise ValueError('Use the local http://127.0.0.1:PORT/api/assistant address')
        if not isinstance(token,str) or not 24<=len(token)<=256 or not re.fullmatch(r'[a-zA-Z0-9_-]+',token):
            raise ValueError('A scoped assistant token is required')
        self.api_base=api_base.rstrip('/')
        self.origin=f'http://127.0.0.1:{parsed.port}'
        self._token=token

    def _send(self,url,payload=None,image=False):
        headers={'X-Assistant-Token':self._token}
        data=None
        if payload is not None:
            headers['Content-Type']='application/json'
            data=json.dumps(payload,allow_nan=False).encode('utf-8')
        try:
            with urlopen(Request(url,data=data,headers=headers),timeout=15) as response:
                body=response.read(8*1024*1024+1)
                if len(body)>8*1024*1024:
                    raise ValueError('Assistant response is too large')
        except HTTPError as error:
            try:
                message=json.loads(error.read(8192)).get('error','Assistant request failed')
            except (ValueError,AttributeError):
                message='Assistant request failed'
            raise ValueError(f'{error.code}: {message}') from None
        return body if image else json.loads(body)

    def request(self,route,payload=None):
        if route not in ('/status','/heartbeat','/frames','/boxes'):
            raise ValueError('Unsupported scoped assistant operation')
        return self._send(self.api_base+route,payload)

    def image(self,route):
        prefix='/api/assistant/image/'
        if not isinstance(route,str) or not route.startswith(prefix) or not FRAME_ID.fullmatch(route[len(prefix):]):
            raise ValueError('Use a frame image URL from the scoped queue')
        image=self._send(self.origin+route,image=True)
        if not image.startswith(b'\x89PNG\r\n\x1a\n'):
            raise ValueError('Expected a PNG review frame')
        return image


def agent_boxes(reply):
    if not isinstance(reply,dict) or type(reply.get('image_viewed')) is not bool or not isinstance(reply.get('boxes'),list) or len(reply['boxes'])>128:
        raise ValueError('Agent must report image_viewed and at most 128 boxes')
    boxes=[]
    for box in reply['boxes']:
        if not isinstance(box,dict) or type(box.get('class_id')) is not int or box['class_id'] not in (0,1):
            raise ValueError('Agent boxes must use class 0 Head or 1 Body')
        values=[box.get(key) for key in ('x1','y1','x2','y2','score')]
        if any(type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1 for value in values) or values[0]>=values[2] or values[1]>=values[3]:
            raise ValueError('Agent boxes must be finite, normalized positive rectangles')
        if values[4]>=.6 and reply['image_viewed']:
            boxes.append({key:box[key] for key in ('class_id','x1','y1','x2','y2','score')})
    return boxes


class AntigravityRunner:
    def __init__(self,cache,*,executable=None,model='gemini-3.8-flash-medium',run=None):
        self.cache=Path(cache).resolve()
        self.model=model
        if not re.fullmatch(r'[a-zA-Z0-9_.-]{1,80}',model):
            raise ValueError('Invalid Antigravity model name')
        self.executable=executable or shutil.which('agy') or str(Path(os.environ.get('LOCALAPPDATA',''))/'agy'/'bin'/'agy.exe')
        if not executable and not Path(self.executable).is_file():
            raise ValueError('Install and sign in to the Antigravity CLI first')
        self.run=run or subprocess.run
        self.total_tokens=0

    def __call__(self,image,frame):
        workspace=Path(image).resolve().parent
        if not workspace.is_relative_to(self.cache):
            raise ValueError('Image review must stay inside its assistant cache')
        schema={'type':'object','properties':{'image_viewed':{'type':'boolean'},'boxes':{'type':'array','maxItems':128,'items':{'type':'object','properties':{'class_id':{'type':'integer','minimum':0,'maximum':1},**{key:{'type':'number','minimum':0,'maximum':1} for key in ('x1','y1','x2','y2','score')}},'required':['class_id','x1','y1','x2','y2','score'],'additionalProperties':False}},'notes':{'type':'string'}},'required':['image_viewed','boxes','notes'],'additionalProperties':False}
        schema_path=workspace/'box-schema.json'
        schema_path.write_text(json.dumps(schema),encoding='utf-8')
        prompt=(f'Inspect the actual image file at {Path(image).resolve()} using your image viewing tool. This is a 640 by 448 game crop. '
                'Only read that image; do not use the terminal, browser, network, write files, create tasks or invoke other agents. '
                'Find clearly visible missing Head (class_id 0) and Body (class_id 1) rectangles. Existing boxes are supplied below: add only missing targets/parts. '
                'Use tight axis-aligned boxes around visible heads and visible body extents. Do not guess hidden heads, invent targets, duplicate existing boxes, or label flashes/weapons as heads. '
                'Return normalized x1,y1,x2,y2 in 0..1 relative to the complete 640x448 image, with a 0..1 score. Include only additions you can confidently locate (score>=0.6). '
                'If you cannot actually view the image, set image_viewed=false and boxes=[]. Return the schema JSON only. '
                f'Existing boxes: {json.dumps(frame.get("boxes",[]))}')
        command=[self.executable,'--log-file',str(workspace/'agent.log'),'--model',self.model,'--mode','plan','--sandbox',
                 '--print-timeout','90s','--output-format','json','--json-schema',str(schema_path),'--print',prompt]
        result=self.run(command,cwd=str(workspace),capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=105,
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if result.returncode:
            raise ValueError('Antigravity could not complete image review; check its sign-in and available quota')
        if len(result.stdout)>2*1024*1024:
            raise ValueError('Antigravity response is too large')
        envelope=json.loads(result.stdout)
        if envelope.get('status')!='SUCCESS':
            raise ValueError('Antigravity image review did not succeed')
        usage=envelope.get('usage',{})
        count=usage.get('total_tokens',0)
        if type(count) is int and count>=0:
            self.total_tokens+=count
        reply=envelope.get('structured_output') or json.loads(envelope.get('response','{}'))
        agent_boxes(reply)
        return reply


def process_frames(client,runner,cache_root,*,frame_limit=4,cursor=0):
    if type(frame_limit) is not int or not 1<=frame_limit<=32 or type(cursor) is not int or cursor<0:
        raise ValueError('Review 1..32 frames from a nonnegative queue cursor')
    status=client.request('/status')
    connection=status['connection']['id']
    if not isinstance(connection,str) or not re.fullmatch(r'assistant-[a-f0-9]{16,64}',connection):
        raise ValueError('Unknown assistant connection')
    root=Path(cache_root).resolve()
    cache=root/connection
    if cache.resolve().parent!=root or cache.is_symlink():
        raise ValueError('Assistant cache must stay in its connection folder')
    cache.mkdir(parents=True,exist_ok=True)
    summary={'processed':0,'added':0,'queued':0,'errors':[],'next_cursor':cursor}
    stopped=threading.Event()
    def heartbeat():
        while not stopped.wait(20):
            try:
                client.request('/heartbeat',{})
            except (ValueError,OSError):
                return
    client.request('/heartbeat',{})
    thread=threading.Thread(target=heartbeat,daemon=True)
    thread.start()
    seen=set()
    try:
        while summary['processed']<frame_limit:
            page=client.request('/frames',{'limit':min(16,frame_limit-summary['processed']),'cursor':cursor})
            frames=page.get('frames',[])
            if not isinstance(frames,list):
                raise ValueError('Invalid frame queue')
            for frame in frames:
                if summary['processed']>=frame_limit:
                    break
                frame_id=frame.get('id')
                revision=frame.get('revision')
                if not isinstance(frame_id,str) or not FRAME_ID.fullmatch(frame_id) or type(revision) is not int or revision<0 or frame_id in seen:
                    raise ValueError('Invalid or repeated queue frame')
                seen.add(frame_id)
                try:
                    image=cache/f'{frame_id}-r{revision}.png'
                    if image.is_symlink() or image.resolve().parent!=cache.resolve():
                        raise ValueError('Unsafe cached image path')
                    image.write_bytes(client.image(frame['image_url']))
                    reply=runner(image,frame)
                    boxes=agent_boxes(reply)
                    if boxes:
                        request_id=hashlib.sha256(f'{connection}:{frame_id}:{revision}'.encode()).hexdigest()
                        result=client.request('/boxes',{'frame_id':frame_id,'revision':revision,'boxes':boxes,'request_id':request_id})
                        summary['added']+=result.get('added',0)
                        summary['queued']+=int(result.get('state')=='queued')
                    if not reply['image_viewed']:
                        summary['errors'].append({'frame_id':frame_id,'error':'Antigravity could not view this image'})
                except (ValueError,OSError,subprocess.SubprocessError) as error:
                    summary['errors'].append({'frame_id':frame_id,'error':str(error)[:300]})
                summary['processed']+=1
                client.request('/heartbeat',{})
            next_cursor=page.get('next_cursor')
            summary['next_cursor']=next_cursor
            if not frames or next_cursor is None:
                break
            if type(next_cursor) is not int or next_cursor<=cursor:
                raise ValueError('Invalid queue cursor')
            cursor=next_cursor
    finally:
        stopped.set()
        thread.join(timeout=1)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description='Review a bounded set of pending frames through the local assistant API')
    parser.add_argument('--api-base')
    parser.add_argument('--token')
    parser.add_argument('--connection-file',type=Path)
    parser.add_argument('--frames',type=int,default=4)
    parser.add_argument('--cursor',type=int,default=0)
    parser.add_argument('--model',default='gemini-3.8-flash-medium')
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args(argv)
    try:
        if args.connection_file:
            if args.api_base or args.token:
                raise ValueError('Use either a connection file or API address/token arguments')
            config=json.loads(args.connection_file.read_text(encoding='utf-8-sig'))
            args.api_base=config['api_base']
            args.token=config['headers']['X-Assistant-Token']
        if not args.api_base or not args.token:
            raise ValueError('Provide --connection-file or both --api-base and --token')
        client=AssistantClient(args.api_base,args.token)
        if args.check:
            result=client.request('/heartbeat',{})
        else:
            cache_root=PROJECT/'runtime-data'/'assistant-cache'
            if not cache_root.resolve().is_relative_to(PROJECT.resolve()):
                raise ValueError('Assistant cache must stay inside the application')
            cache_root.mkdir(parents=True,exist_ok=True)
            runner=AntigravityRunner(cache_root,model=args.model)
            result=process_frames(client,runner,cache_root,frame_limit=args.frames,cursor=args.cursor)
            result['tokens_used']=runner.total_tokens
        print(json.dumps(result,ensure_ascii=False,allow_nan=False))
        return 0
    except (ValueError,OSError,KeyError,TypeError,subprocess.SubprocessError) as error:
        print(f'Assistant bridge: {error}',file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
