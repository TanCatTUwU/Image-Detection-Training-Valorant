from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import threading
import numpy as np
from prepare_dataset import ONNXProposer

@lru_cache(maxsize=1)
def hardware():
    import onnxruntime as ort
    result = {'device': None, 'label': None, 'providers': ort.get_available_providers()}
    try:
        import torch
        device = 'cuda:0' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else None
        if device:
            probe = torch.ones(1, device=device)
            if float(probe.cpu()[0]) == 1:
                result.update(device=device, label='Ultralytics · ' + (torch.cuda.get_device_name(0) if device == 'cuda:0' else 'Apple GPU'))
    except (ImportError, OSError, RuntimeError, AssertionError):
        pass
    return result

def choose_inference(model_path, capabilities=None):
    capabilities = hardware() if capabilities is None else capabilities
    path = Path(model_path)
    checkpoint = path.with_name('checkpoint.pt' if path.name == 'model.onnx' else 'best.pt')
    if capabilities.get('device') and checkpoint.is_file() and not checkpoint.is_symlink() and checkpoint.resolve().parent == path.parent.resolve():
        return dict(backend='ultralytics', device=capabilities['device'], label=capabilities.get('label') or 'Ultralytics · GPU', checkpoint=str(checkpoint))
    providers = capabilities.get('providers', [])
    if 'DmlExecutionProvider' in providers:
        return dict(backend='directml', device=0, label='DirectML · GPU')
    if 'CUDAExecutionProvider' in providers:
        return dict(backend='onnx-cuda', device=0, label='ONNX · CUDA GPU')
    return dict(backend='onnx-cpu', device='cpu', label='ONNX · CPU')

class HardwareMonitor:

    def __init__(self):
        self._lock = threading.Lock()
        self._started = False
        self._value = None

    def _load(self):
        try:
            value = hardware()
        except Exception:
            value = {'device': None, 'providers': [], 'error': 'Hardware check unavailable'}
        with self._lock:
            self._value = value

    def snapshot(self):
        with self._lock:
            if not self._started:
                self._started = True
                threading.Thread(target=self._load, daemon=True).start()
            return dict(self._value) if self._value is not None else None

class UltralyticsProposer(ONNXProposer):

    def __init__(self, model_path, choice, confidence, tiles, strict=False):
        from model_registry import validate_onnx
        from ultralytics import YOLO
        shape = validate_onnx(model_path)['input_shape']
        self.model = YOLO(choice['checkpoint'], task='detect')
        if self.model.task != 'detect' or self.model.names != {0: 'Head', 1: 'Body'}:
            raise ValueError('Annotation model must use classes exactly 0=Head, 1=Body')
        self.device = choice['device']
        self.model.to(self.device)
        self.input_size = shape[2:]
        self.confidence, self.tiles, self.strict = confidence, tiles, strict

    def _pass(self, frame):
        result = self.model.predict(source=frame, device=self.device, imgsz=self.input_size, conf=self.confidence, iou=.65, max_det=300, rect=False, verbose=False, save=False)[0]
        if result.boxes is None:
            if self.strict:
                raise ValueError('The model did not return detection boxes')
            return []
        coords = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        if coords.ndim != 2 or coords.shape[1] != 4 or classes.shape != (len(coords),) or scores.shape != (len(coords),) or not all(np.isfinite(value).all() for value in (coords, classes, scores)):
            raise ValueError('Model output contains invalid detections')
        if not np.isin(classes, [0, 1]).all() or not ((scores >= 0) & (scores <= 1)).all():
            raise ValueError('Model output contains invalid classes or confidence')
        height, width = frame.shape[:2]
        normalized = np.clip(coords / [width, height, width, height], 0, 1)
        return [dict(class_id=int(class_id), x1=float(box[0]), y1=float(box[1]), x2=float(box[2]), y2=float(box[3]), score=float(score))
                for box, class_id, score in zip(normalized, classes, scores) if box[2] > box[0] and box[3] > box[1]]

class RuntimeProposer(ONNXProposer):

    def __init__(self, model_path, choice, confidence, tiles, strict=False):
        import onnxruntime as ort
        from model_registry import validate_onnx
        shape = validate_onnx(model_path)['input_shape']
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        options.add_session_config_entry('session.inter_op.allow_spinning', '0')
        provider = 'CUDAExecutionProvider' if choice['backend'] == 'onnx-cuda' else 'CPUExecutionProvider'
        session = ort.InferenceSession(str(model_path), sess_options=options, providers=[provider])
        if session.get_providers()[0] != provider:
            raise RuntimeError('The selected GPU runtime could not start')
        session.disable_fallback()
        self.model = SimpleNamespace(session=session, input_name=session.get_inputs()[0].name, input_size=(shape[3], shape[2]), names={0: 'Head', 1: 'Body'}, end_to_end=False)
        self.confidence, self.tiles, self.strict = confidence, tiles, strict

def create_proposer(model_path, device_id=0, confidence=.25, tiles=True, *, strict=False):
    try:
        choice = choose_inference(model_path)
        if choice['backend'] == 'ultralytics':
            return UltralyticsProposer(model_path, choice, confidence, tiles, strict)
        if choice['backend'] == 'directml':
            return ONNXProposer(model_path, device_id, confidence, tiles, strict=strict)
        return RuntimeProposer(model_path, choice, confidence, tiles, strict)
    except (ValueError, FileNotFoundError):
        raise
    except Exception as exc:
        raise RuntimeError(f'Model initialization failed: {str(exc)[:400]}') from exc

def validate_runtime(model_path):
    from model_registry import validate_onnx
    shape = validate_onnx(model_path)['input_shape']
    proposer = create_proposer(model_path, tiles=False, strict=True)
    try:
        proposer._pass(np.zeros((shape[2], shape[3], 3), dtype=np.uint8))
    except Exception as exc:
        raise ValueError(f'The model cannot run with the selected backend: {str(exc)[:400]}') from exc
    finally:
        session = getattr(getattr(proposer, 'model', None), 'session', None)
        if session is not None and hasattr(session, 'close'):
            session.close()
    return choose_inference(model_path)
