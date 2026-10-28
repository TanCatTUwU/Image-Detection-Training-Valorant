from __future__ import annotations
from collections import deque
import json
from pathlib import Path
import queue
import subprocess
import threading
from types import SimpleNamespace
from assistant_loop import ProviderSessionEnded


class PersistentCLI:
    def __init__(self, *, popen=None):
        self.popen=popen or subprocess.Popen
        self.process=None
        self.identity=None
        self.events=queue.Queue()
        self.errors=deque(maxlen=16)
        self.tokens=0
        self.closed=False
        self.inflight=False
        self.readers=[]
        self.call_lock=threading.Lock()

    def _read_output(self):
        try:
            while True:
                line=self.process.stdout.readline(2*1024*1024+1)
                if not line:
                    break
                if len(line)>2*1024*1024:
                    raise ValueError('Antigravity stream response is too large')
                event=json.loads(line)
                if not isinstance(event,dict):
                    raise ValueError('Invalid Antigravity stream event')
                if event.get('event')=='result':
                    self.events.put(('result',event.get('result')))
        except OSError:
            self.events.put(('eof',None))
        except ValueError as error:
            self.events.put(('error',str(error)))
        finally:
            self.events.put(('eof',None))

    def _read_errors(self):
        try:
            while True:
                line=self.process.stderr.readline(8193)
                if not line:
                    break
                self.errors.append(line[-8192:])
        except (ValueError,OSError):
            pass

    def __call__(self,command,*,cwd,timeout,**kwargs):
        if not self.call_lock.acquire(blocking=False):
            raise ValueError('The Antigravity CLI is already reviewing a batch')
        try:
            return self._review(command,cwd=cwd,timeout=timeout,**kwargs)
        finally:
            self.call_lock.release()

    def _review(self,command,*,cwd,timeout,**kwargs):
        if self.closed:
            raise ProviderSessionEnded('The Antigravity CLI session ended')
        arguments=list(command)
        prompt_index=arguments.index('--print')
        if prompt_index!=len(arguments)-2:
            raise ValueError('Expected one Antigravity review prompt')
        prompt=arguments.pop();arguments.pop()
        arguments[arguments.index('--output-format')+1]='stream-json'
        arguments[arguments.index('--print-timeout')+1]='240s'
        arguments+=['--input-format','stream-json']
        identity=(tuple(arguments),str(Path(cwd).resolve()))
        if self.identity is not None and identity!=self.identity:
            raise ValueError('The approved Antigravity command or workspace changed')
        if self.process is None:
            self.identity=identity
            self.process=self.popen(arguments,cwd=cwd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                    text=True,encoding='utf-8',errors='replace',bufsize=1,
                                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            self.readers=[threading.Thread(target=self._read_output,daemon=True),threading.Thread(target=self._read_errors,daemon=True)]
            for reader in self.readers:
                reader.start()
        if self.process.poll() is not None:
            self.close()
            raise ProviderSessionEnded('The Antigravity CLI session exited')
        self.inflight=True
        self.errors.clear()
        def send():
            try:
                self.process.stdin.write(json.dumps(dict(event='user',message=dict(content=prompt)),allow_nan=False)+'\n')
                self.process.stdin.flush()
            except (ValueError,OSError) as error:
                self.events.put(('eof',None))
        threading.Thread(target=send,daemon=True).start()
        try:
            event,value=self.events.get(timeout=timeout)
            if event=='eof':
                raise ProviderSessionEnded('Antigravity session ended without a valid result')
            if event!='result' or not isinstance(value,dict):
                raise ValueError('Invalid Antigravity stream result')
            usage=value.get('usage') or {}
            count=usage.get('total_tokens',0) if isinstance(usage,dict) else 0
            if type(count) is not int or count<0:
                raise ValueError('Invalid cumulative Antigravity usage')
            value=dict(value,usage=dict(total_tokens=max(0,count-self.tokens)))
            self.tokens=max(self.tokens,count)
            self.inflight=False
            return SimpleNamespace(returncode=0,stdout=json.dumps(value),stderr=''.join(self.errors)[-8192:])
        except queue.Empty:
            self.close()
            raise subprocess.TimeoutExpired(arguments,timeout) from None
        except (ValueError,OSError,KeyboardInterrupt):
            self.close()
            raise

    def close(self):
        self.closed=True
        if self.process is None:
            return
        process=self.process
        try:
            if self.inflight and process.poll() is None:
                process.terminate()
                process.wait(timeout=3)
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            if process.poll() is None:
                process.wait(timeout=5)
        except (OSError,ValueError,subprocess.TimeoutExpired):
            if process.poll() is None:
                process.kill()
                process.wait(timeout=3)
        finally:
            for reader,stream in zip(self.readers,(process.stdout,process.stderr)):
                reader.join(timeout=.2)
                try:
                    if not reader.is_alive():
                        stream.close()
                except (OSError,ValueError):
                    pass
