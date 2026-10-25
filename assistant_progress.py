from __future__ import annotations
import json
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path


def validate_event(payload, frame_pattern):
    if not isinstance(payload, dict) or set(payload) - {'event', 'batch_id', 'frames', 'total', 'error'}:
        raise ValueError('Expected a bounded assistant progress event')
    event, batch_id = payload.get('event'), payload.get('batch_id')
    if event not in ('start', 'complete', 'fail', 'finish') or not isinstance(batch_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:-]{0,79}', batch_id):
        raise ValueError('Expected a progress event and safe batch_id of 1..80 characters')
    if 'total' in payload and (event != 'start' or type(payload['total']) is not int or not 1 <= payload['total'] <= 32):
        raise ValueError('Only start accepts an integer total of 1..32')
    if 'error' in payload:
        error = payload['error']
        if event != 'fail' or not isinstance(error, str) or not 1 <= len(error) <= 500 or any(ord(char) < 32 or 127 <= ord(char) < 160 for char in error):
            raise ValueError('Only fail accepts a single-line error of 1..500 characters')
    if event == 'finish':
        if set(payload) != {'event', 'batch_id'}:
            raise ValueError('Finish accepts only event and batch_id')
        return event, batch_id, [], None, None
    frames = payload.get('frames')
    if not isinstance(frames, list) or not 1 <= len(frames) <= 32:
        raise ValueError('Progress requires 1..32 unique frames')
    clean, seen = [], set()
    for frame in frames:
        if not isinstance(frame, dict) or set(frame) != {'frame_id', 'revision'}:
            raise ValueError('Progress frames require frame_id and revision')
        fid, revision = frame['frame_id'], frame['revision']
        if not isinstance(fid, str) or not frame_pattern.fullmatch(fid) or type(revision) is not int or revision < 0 or fid in seen:
            raise ValueError('Progress requires unique valid frame IDs and non-negative revisions')
        seen.add(fid)
        clean.append((fid, revision))
    if event == 'start' and payload.get('total', 32) < len(clean):
        raise ValueError('Batch total cannot be smaller than its frames')
    return event, batch_id, clean, payload.get('total'), payload.get('error')


class ProgressStore:
    def __init__(self, project):
        self.project = Path(project).resolve()
        self.path = self.project / 'runtime-data' / 'assistant-progress.sqlite3'

    @contextmanager
    def transaction(self):
        parent = self.path.parent
        if parent.is_symlink() or parent.resolve().parent != self.project or self.path.is_symlink():
            raise ValueError('Assistant progress must stay inside project runtime-data')
        parent.mkdir(exist_ok=True)
        database = None
        try:
            database = sqlite3.connect(self.path, timeout=5)
            database.row_factory = sqlite3.Row
            database.executescript('''
                CREATE TABLE IF NOT EXISTS frame_progress (
                    connection_id TEXT NOT NULL,
                    frame_id TEXT NOT NULL,
                    completed_revision INTEGER,
                    ever_completed INTEGER NOT NULL DEFAULT 0,
                    ever_failed INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(connection_id, frame_id)
                );
                CREATE TABLE IF NOT EXISTS batches (
                    connection_id TEXT NOT NULL,
                    batch_id TEXT NOT NULL,
                    items TEXT NOT NULL,
                    total INTEGER NOT NULL,
                    finished INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(connection_id, batch_id)
                );
                CREATE TABLE IF NOT EXISTS connection_progress (
                    connection_id TEXT PRIMARY KEY,
                    batch_id TEXT,
                    last_completed_frame TEXT,
                    updated_at TEXT
                );
                CREATE TABLE IF NOT EXISTS progress_scopes (
                    connection_id TEXT PRIMARY KEY,
                    recording_id TEXT NOT NULL,
                    provider TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS progress_scope_recording ON progress_scopes(recording_id,provider);
            ''')
            database.execute('BEGIN IMMEDIATE')
            yield database
            database.commit()
        except sqlite3.Error as exc:
            if database is not None:
                database.rollback()
            raise RuntimeError('Could not read or save local assistant progress') from exc
        except BaseException:
            if database is not None:
                database.rollback()
            raise
        finally:
            if database is not None:
                database.close()

    def batch(self, database, connection_id, batch_id):
        row = database.execute('SELECT * FROM batches WHERE connection_id=? AND batch_id=?', (connection_id, batch_id)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result['items'] = json.loads(row['items'])
        return result

    def active_batch(self, database, connection_id):
        row = database.execute('SELECT batch_id FROM connection_progress WHERE connection_id=?', (connection_id,)).fetchone()
        return row['batch_id'] if row else None

    def save_batch(self, database, connection_id, batch, now, *, activate=False):
        database.execute('''INSERT INTO batches(connection_id,batch_id,items,total,finished,updated_at) VALUES(?,?,?,?,?,?)
            ON CONFLICT(connection_id,batch_id) DO UPDATE SET items=excluded.items,total=excluded.total,finished=excluded.finished,updated_at=excluded.updated_at''',
            (connection_id, batch['batch_id'], json.dumps(batch['items'], separators=(',', ':')), batch['total'], int(batch['finished']), now))
        if activate:
            database.execute('''INSERT INTO connection_progress(connection_id,batch_id,updated_at) VALUES(?,?,?)
                ON CONFLICT(connection_id) DO UPDATE SET batch_id=excluded.batch_id,updated_at=excluded.updated_at''', (connection_id, batch['batch_id'], now))
        else:
            database.execute('UPDATE connection_progress SET updated_at=? WHERE connection_id=? AND batch_id=?', (now, connection_id, batch['batch_id']))

    def success(self, database, connection_id, frame_id, revision, now):
        old = database.execute('SELECT * FROM frame_progress WHERE connection_id=? AND frame_id=?', (connection_id, frame_id)).fetchone()
        if old is not None and old['ever_completed'] and old['completed_revision'] >= revision:
            return
        database.execute('''INSERT INTO frame_progress(connection_id,frame_id,completed_revision,ever_completed) VALUES(?,?,?,1)
            ON CONFLICT(connection_id,frame_id) DO UPDATE SET completed_revision=MAX(COALESCE(frame_progress.completed_revision,-1),excluded.completed_revision),ever_completed=1''', (connection_id, frame_id, revision))
        database.execute('''INSERT INTO connection_progress(connection_id,last_completed_frame,updated_at) VALUES(?,?,?)
            ON CONFLICT(connection_id) DO UPDATE SET last_completed_frame=excluded.last_completed_frame,updated_at=excluded.updated_at''', (connection_id, frame_id, now))

    def failure(self, database, connection_id, frame_id):
        database.execute('''INSERT INTO frame_progress(connection_id,frame_id,ever_failed) VALUES(?,?,1)
            ON CONFLICT(connection_id,frame_id) DO UPDATE SET ever_failed=1''', (connection_id, frame_id))

    def completed(self, connection_id):
        with self.transaction() as database:
            return {row['frame_id']: row['completed_revision'] for row in database.execute('SELECT frame_id,completed_revision FROM frame_progress WHERE connection_id=? AND ever_completed=1', (connection_id,))}

    def bind_scopes(self, scopes):
        if not scopes:
            return
        with self.transaction() as database:
            for connection_id,recording_id,provider in scopes:
                if not re.fullmatch(r'assistant-[a-f0-9]{24}',connection_id) or not re.fullmatch(r'run-[a-f0-9]{24}',recording_id) or provider not in ('antigravity','custom'):
                    raise ValueError('Invalid recording progress scope')
                old=database.execute('SELECT recording_id,provider FROM progress_scopes WHERE connection_id=?',(connection_id,)).fetchone()
                if old and (old['recording_id'],old['provider'])!=(recording_id,provider):
                    raise ValueError('A saved assistant progress scope cannot change')
                database.execute('INSERT OR IGNORE INTO progress_scopes(connection_id,recording_id,provider) VALUES(?,?,?)',(connection_id,recording_id,provider))

    def recording_completed(self, recording_id, provider):
        with self.transaction() as database:
            return {row['frame_id']:row['revision'] for row in database.execute('''
                SELECT progress.frame_id,MAX(progress.completed_revision) AS revision FROM frame_progress AS progress
                JOIN progress_scopes AS scope ON progress.connection_id=scope.connection_id
                WHERE scope.recording_id=? AND scope.provider=? AND progress.ever_completed=1 GROUP BY progress.frame_id''',(recording_id,provider))}

    def recording_last(self, recording_id, provider):
        with self.transaction() as database:
            row=database.execute('''SELECT progress.last_completed_frame FROM connection_progress AS progress
                JOIN progress_scopes AS scope ON progress.connection_id=scope.connection_id
                WHERE scope.recording_id=? AND scope.provider=? AND progress.last_completed_frame IS NOT NULL
                ORDER BY progress.updated_at DESC,scope.connection_id DESC LIMIT 1''',(recording_id,provider)).fetchone()
            return row['last_completed_frame'] if row else None

    def snapshot(self, connection_id, frames):
        with self.transaction() as database:
            connection = database.execute('SELECT * FROM connection_progress WHERE connection_id=?', (connection_id,)).fetchone()
            counts = database.execute('SELECT COALESCE(SUM(ever_completed),0),COALESCE(SUM(ever_failed),0) FROM frame_progress WHERE connection_id=?', (connection_id,)).fetchone()
            batch = self.batch(database, connection_id, connection['batch_id']) if connection and connection['batch_id'] else None
            result = dict(state='idle', current_frames=[], last_completed_frame=connection['last_completed_frame'] if connection else None,
                          processed=counts[0], failed=counts[1], batch_id=batch['batch_id'] if batch else None,
                          batch_total=batch['total'] if batch else 0, batch_completed=0, batch_failed=0,
                          updated_at=connection['updated_at'] if connection else None)
            if batch:
                result['batch_completed'] = sum(item['state'] == 'completed' for item in batch['items'].values())
                result['batch_failed'] = sum(item['state'] == 'failed' for item in batch['items'].values())
                for fid, item in batch['items'].items():
                    frame = frames.get(fid)
                    if item['state'] == 'processing' and frame and frame['status'] == 'pending' and frame['revision'] in (item['revision'], item.get('completed_revision')):
                        result['current_frames'].append(dict(frame_id=fid, revision=item['revision'], started_at=item['started_at']))
                result['state'] = 'failed' if batch['finished'] and (result['batch_failed'] or any(item['state'] == 'processing' for item in batch['items'].values())) else ('completed' if batch['finished'] else 'processing')
            return result
