from __future__ import annotations
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import numpy as np


def build_screening_model(source, destination):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from model_registry import validate_onnx
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise ValueError('The screening graph must have its own temporary file')
    shape = validate_onnx(source)['input_shape']
    model = onnx.load(source, load_external_data=False)
    output = model.graph.output[0].name
    version = next((item.version for item in model.opset_import if item.domain in ('', 'ai.onnx')), 0)
    if not 11 <= version <= 21:
        raise ValueError('GPU screening requires ONNX operator version 11 through 21')
    names = {value for node in model.graph.node for value in (*node.input, *node.output)}
    names.update(item.name for item in model.graph.initializer)
    prefix = 'empty_screen_'
    while any(name.startswith(prefix) for name in names):
        prefix += 'x_'
    def name(value):
        return prefix + value
    model.graph.initializer.extend([
        numpy_helper.from_array(np.array([4, 5], dtype=np.int64), name('classes')),
    ])
    if version >= 18:
        model.graph.initializer.append(numpy_helper.from_array(np.array([1, 2], dtype=np.int64), name('axes')))
    def reduce(value, result):
        return helper.make_node('ReduceMax', [value, name('axes')], [result], keepdims=1) if version >= 18 else helper.make_node('ReduceMax', [value], [result], axes=[1, 2], keepdims=1)
    model.graph.node.extend([
        helper.make_node('Gather', [output, name('classes')], [name('scores')], axis=1),
        reduce(name('scores'), name('maximum')),
        helper.make_node('IsNaN', [output], [name('nan')]),
        helper.make_node('IsInf', [output], [name('inf')]),
        helper.make_node('Or', [name('nan'), name('inf')], [name('invalid')]),
        helper.make_node('Cast', [name('invalid')], [name('invalid_float')], to=TensorProto.FLOAT),
        reduce(name('invalid_float'), name('invalid_maximum')),
        helper.make_node('Concat', [name('maximum'), name('invalid_maximum')], [name('summary')], axis=1),
    ])
    del model.graph.output[:]
    model.graph.output.append(helper.make_tensor_value_info(name('summary'), TensorProto.FLOAT, [1, 2, 1]))
    onnx.checker.check_model(model)
    onnx.save_model(model, destination)
    return shape


class _ScreeningSession:

    def __init__(self, session, temporary):
        self.session, self.temporary = session, temporary

    def run(self, *args):
        if self.session is None:
            raise RuntimeError('The GPU screening session is closed')
        return self.session.run(*args)

    def close(self):
        session, self.session = self.session, None
        try:
            if callable(getattr(session, 'close', None)):
                session.close()
        finally:
            self.temporary.cleanup()


class GPUScreeningProposer:

    def __init__(self, model_path, device_id=0, confidence=.08, tiles=True, *, strict=True):
        import onnxruntime as ort
        if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 < confidence <= 1 or not strict:
            raise ValueError('GPU screening requires strict detection and a finite confidence in 0..1')
        if type(device_id) is not int or device_id < 0:
            raise ValueError('Choose a nonnegative GPU device ID')
        providers = ort.get_available_providers()
        provider = next((value for value in ('DmlExecutionProvider', 'CUDAExecutionProvider') if value in providers), None)
        if provider is None:
            raise RuntimeError('Empty-frame screening requires a DirectML or CUDA GPU runtime; CPU fallback is disabled')
        import cv2
        cv2.setNumThreads(1)
        temporary = tempfile.TemporaryDirectory(prefix='task-gpu-scan-')
        session = None
        try:
            path = Path(temporary.name) / 'screen.onnx'
            shape = build_screening_model(model_path, path)
            if provider == 'DmlExecutionProvider':
                from native_dml import NativeDMLSession
                session = NativeDMLSession(path, device_id)
                self.execution = dict(backend='directml', device=device_id, label='DirectML · GPU')
            else:
                options = ort.SessionOptions()
                options.intra_op_num_threads = options.inter_op_num_threads = 1
                options.add_session_config_entry('session.intra_op.allow_spinning', '0')
                options.add_session_config_entry('session.inter_op.allow_spinning', '0')
                options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
                session = ort.InferenceSession(str(path), sess_options=options, providers=[(provider, {'device_id': device_id})])
                session.disable_fallback()
                self.execution = dict(backend='onnx-cuda', device=device_id, label='ONNX · CUDA GPU')
            if session.get_providers()[0] != provider or session.get_outputs()[0].shape != [1, 2, 1]:
                raise RuntimeError('The GPU screening graph did not start on the requested device')
            self.model = SimpleNamespace(session=_ScreeningSession(session, temporary), input_name=session.get_inputs()[0].name, input_size=(shape[3], shape[2]))
            self.confidence, self.tiles, self.strict = confidence, tiles, strict
        except BaseException:
            if callable(getattr(session, 'close', None)):
                session.close()
            temporary.cleanup()
            raise

    def contains_targets(self, frame):
        from directml_backend import prepare_frame
        tensor, _ = prepare_frame(frame, self.model.input_size)
        summary = np.asarray(self.model.session.run(None, {self.model.input_name: tensor})[0])
        if summary.shape != (1, 2, 1) or summary.dtype != np.float32 or not np.isfinite(summary).all() or summary[0, 1, 0] != 0:
            raise ValueError('Strict GPU screening returned invalid model output; the frame was preserved')
        return bool(summary[0, 0, 0] >= max(0, self.confidence - 1e-5))


def create_screening_proposer(model_path, device_id=0, confidence=.08, tiles=True, *, strict=True):
    return GPUScreeningProposer(model_path, device_id, confidence, tiles, strict=strict)
