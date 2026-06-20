"""MT3 (Google Magenta) multitrack-transcription driver — isolated heavy backend.

This module is quarantined from :mod:`.transcribe` on purpose: it pulls the full MT3 / T5X / JAX /
TensorFlow stack and the canonical inference glue from the magenta/mt3 reference (the colab
``InferenceModel``). Nothing heavy is imported at module load — every dependency is imported
lazily inside :func:`transcribe_mt3`, so importing this file is cheap and the deterministic core
(and its tests) never drag JAX/TF in.

Why a vendored ``InferenceModel`` instead of a library call: the magenta/mt3 package ships the
model + gin configs but no stable ``audio -> NoteSequence`` entry point; the reference path is the
colab class reproduced below, adapted to discover its gin files from the installed package and to
take the checkpoint directory from ``AUDIO2AY3_MT3_CHECKPOINT``.

Setup (on a machine with a working CPU/GPU; see the ``[mt3]`` extra):

    pip install -e ".[mt3]"
    # download a checkpoint, e.g. the "mt3" multi-instrument checkpoint from gs://mt3/checkpoints
    setx AUDIO2AY3_MT3_CHECKPOINT  C:\\path\\to\\mt3\\checkpoint

The output is a :class:`note_seq.NoteSequence`; :func:`audio2ay3.analysis.transcribe`
.note_sequence_to_transcription turns it into the neutral IR.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import numpy as np

# MT3 operates on 16 kHz mono audio regardless of the project render rate.
MT3_SAMPLE_RATE = 16000
# Multi-instrument checkpoint by default; "ismir2021" is the piano-only alternative.
_MODEL_TYPE = "mt3"
_CHECKPOINT_ENV = "AUDIO2AY3_MT3_CHECKPOINT"

# Cache the (expensive) model build/restore across calls in one process, keyed by checkpoint path.
_MODEL_CACHE: dict[str, object] = {}


def transcribe_mt3(audio: np.ndarray, sr: int):
    """Transcribe mono ``audio`` (sample rate ``sr``) with MT3, returning a ``NoteSequence``.

    Raises ``RuntimeError`` with actionable guidance when the optional stack or the checkpoint is
    unavailable, rather than surfacing a raw ``ImportError`` deep inside JAX/T5X.
    """
    _require_dependencies()
    model = _get_model(_checkpoint_path())
    wav = _to_mono_16k(audio, sr)
    return model(wav)


def _require_dependencies() -> None:
    # Probe the MT3-only packages first. They are absent on a base install, so this yields a clean
    # RuntimeError without importing TensorFlow (which can be fragile on some machines).
    try:
        import jax  # noqa: F401
        import t5x  # noqa: F401
        from mt3 import models  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise RuntimeError(
            "MT3 transcription needs the 'mt3' extra (T5X + JAX + magenta/mt3): "
            'pip install -e ".[mt3]"  (see audio2ay3/analysis/_mt3_infer.py for setup)'
        ) from exc


def _checkpoint_path() -> str:
    path = os.environ.get(_CHECKPOINT_ENV, "").strip()
    if not path:
        raise RuntimeError(
            f"MT3 needs a model checkpoint; set {_CHECKPOINT_ENV} to the directory of a "
            "downloaded MT3 checkpoint (e.g. the 'mt3' checkpoint from gs://mt3/checkpoints)."
        )
    if not Path(path).exists():
        raise RuntimeError(f"{_CHECKPOINT_ENV} points at a missing path: {path!r}")
    return path


def _gin_dir() -> str:
    """Locate the gin configs shipped inside the installed ``mt3`` package."""
    import mt3

    return str(Path(mt3.__file__).resolve().parent / "gin")


def _to_mono_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    mono = audio if audio.ndim == 1 else np.mean(audio, axis=tuple(range(1, audio.ndim)))
    mono = np.asarray(mono, dtype=np.float32)
    if sr != MT3_SAMPLE_RATE and mono.size:
        import librosa

        mono = librosa.resample(mono, orig_sr=sr, target_sr=MT3_SAMPLE_RATE)
    return np.ascontiguousarray(mono, dtype=np.float32)


def _get_model(checkpoint_path: str):
    model = _MODEL_CACHE.get(checkpoint_path)
    if model is None:
        model = _InferenceModel(checkpoint_path, _MODEL_TYPE)
        _MODEL_CACHE[checkpoint_path] = model
    return model


class _InferenceModel:
    """Wraps a T5X MT3 model for ``audio -> NoteSequence`` inference.

    Adapted from the magenta/mt3 colab reference. Kept faithful to the published pipeline so it
    tracks the upstream checkpoints; only gin-file discovery and checkpoint selection are wired to
    this project. This class is only ever constructed on a machine with the ``[mt3]`` stack
    installed.
    """

    def __init__(self, checkpoint_path: str, model_type: str = "mt3"):
        import seqio
        import t5x
        import tensorflow as tf
        from mt3 import network, note_sequences, spectrograms, vocabularies

        if model_type == "ismir2021":
            num_velocity_bins = 127
            self.encoding_spec = note_sequences.NoteEncodingSpec
            self.inputs_length = 512
        elif model_type == "mt3":
            num_velocity_bins = 1
            self.encoding_spec = note_sequences.NoteEncodingWithTiesSpec
            self.inputs_length = 256
        else:
            raise ValueError(f"unknown MT3 model_type: {model_type!r}")

        gin_files = [
            os.path.join(_gin_dir(), "model.gin"),
            os.path.join(_gin_dir(), f"{model_type}.gin"),
        ]

        self.batch_size = 8
        self.outputs_length = 1024
        self.sequence_length = {"inputs": self.inputs_length, "targets": self.outputs_length}

        self.partitioner = t5x.partitioning.PjitPartitioner(num_partitions=1)

        self.spectrogram_config = spectrograms.SpectrogramConfig()
        self.codec = vocabularies.build_codec(
            vocab_config=vocabularies.VocabularyConfig(num_velocity_bins=num_velocity_bins)
        )
        self.vocabulary = vocabularies.vocabulary_from_codec(self.codec)
        self.output_features = {
            "inputs": seqio.ContinuousFeature(dtype=tf.float32, rank=2),
            "targets": seqio.Feature(vocabulary=self.vocabulary),
        }

        self._parse_gin(gin_files)
        self.model = self._load_model(network, spectrograms)
        self._restore_from_checkpoint(checkpoint_path)

    @property
    def input_shapes(self):
        return {
            "encoder_input_tokens": (self.batch_size, self.inputs_length),
            "decoder_input_tokens": (self.batch_size, self.outputs_length),
        }

    def _parse_gin(self, gin_files):
        import gin

        gin_bindings = [
            "from __gin__ import dynamic_registration",
            "from mt3 import vocabularies",
            "VOCAB_CONFIG=@vocabularies.VocabularyConfig()",
            "vocabularies.VocabularyConfig.num_velocity_bins=%NUM_VELOCITY_BINS",
        ]
        with gin.unlock_config():
            gin.parse_config_files_and_bindings(
                gin_files, gin_bindings, finalize_config=False
            )

    def _load_model(self, network, spectrograms):
        import gin
        import t5x
        from mt3 import models

        model_config = gin.get_configurable(network.T5Config)()
        module = network.Transformer(config=model_config)
        return models.ContinuousInputsEncoderDecoderModel(
            module=module,
            input_vocabulary=self.output_features["inputs"].vocabulary,
            output_vocabulary=self.output_features["targets"].vocabulary,
            optimizer_def=t5x.adafactor.Adafactor(decay_rate=0.8, step_offset=0),
            input_depth=spectrograms.input_depth(self.spectrogram_config),
        )

    def _restore_from_checkpoint(self, checkpoint_path):
        import jax
        import t5x

        train_state_initializer = t5x.utils.TrainStateInitializer(
            optimizer_def=self.model.optimizer_def,
            init_fn=self.model.get_initial_variables,
            input_shapes=self.input_shapes,
            partitioner=self.partitioner,
        )
        restore_checkpoint_cfg = t5x.utils.RestoreCheckpointConfig(
            path=checkpoint_path, mode="specific", dtype="float32"
        )
        train_state_axes = train_state_initializer.train_state_axes
        self._predict_fn = self._get_predict_fn(train_state_axes)
        self._train_state = train_state_initializer.from_checkpoint_or_scratch(
            [restore_checkpoint_cfg], init_rng=jax.random.PRNGKey(0)
        )

    @functools.lru_cache()  # noqa: B019 - one model instance per process; bounded cache is fine
    def _get_predict_fn(self, train_state_axes):
        import t5x

        def partial_predict_fn(params, batch, decode_rng):
            return self.model.predict_batch_with_aux(
                params, batch, decoder_params={"decode_rng": None}
            )

        return self.partitioner.partition(
            partial_predict_fn,
            in_axis_resources=(
                train_state_axes.params,
                t5x.partitioning.PartitionSpec("data"),
                None,
            ),
            out_axis_resources=t5x.partitioning.PartitionSpec("data"),
        )

    def predict_tokens(self, batch, seed=0):
        import jax

        prediction, _ = self._predict_fn(
            self._train_state.params, batch, jax.random.PRNGKey(seed)
        )
        return self.vocabulary.decode_tf(prediction).numpy()

    def __call__(self, audio):
        from mt3 import metrics_utils

        ds = self._audio_to_dataset(audio)
        ds = self._preprocess(ds)
        model_ds = self.model.FEATURE_CONVERTER_CLS(pack=False)(
            ds, task_feature_lengths=self.sequence_length
        )
        model_ds = model_ds.batch(self.batch_size)

        inferences = (
            tokens
            for batch in model_ds.as_numpy_iterator()
            for tokens in self.predict_tokens(batch)
        )

        predictions = []
        for example, tokens in zip(ds.as_numpy_iterator(), inferences):
            predictions.append(self._postprocess(tokens, example))

        result = metrics_utils.event_predictions_to_ns(
            predictions, codec=self.codec, encoding_spec=self.encoding_spec
        )
        return result["est_ns"]

    def _audio_to_dataset(self, audio):
        import tensorflow as tf

        frames, frame_times = self._audio_to_frames(audio)
        return tf.data.Dataset.from_tensors(
            {"inputs": frames, "input_times": frame_times}
        )

    def _audio_to_frames(self, audio):
        from mt3 import spectrograms

        frame_size = self.spectrogram_config.hop_width
        padding = [0, frame_size - len(audio) % frame_size]
        audio = np.pad(audio, padding, mode="constant")
        frames = spectrograms.split_audio(audio, self.spectrogram_config)
        num_frames = len(audio) // frame_size
        times = np.arange(num_frames) / self.spectrogram_config.frames_per_second
        return frames, times

    def _preprocess(self, ds):
        import t5
        from mt3 import preprocessors

        pp_chain = [
            functools.partial(
                t5.data.preprocessors.split_tokens_to_inputs_length,
                sequence_length=self.sequence_length,
                output_features=self.output_features,
                feature_key="inputs",
                additional_feature_keys=["input_times"],
            ),
            preprocessors.add_dummy_targets,
            functools.partial(
                preprocessors.compute_spectrograms,
                spectrogram_config=self.spectrogram_config,
            ),
        ]
        for pp in pp_chain:
            ds = pp(ds)
        return ds

    def _postprocess(self, tokens, example):
        from mt3 import vocabularies

        tokens = self._trim_eos(tokens, vocabularies.DECODED_EOS_ID)
        start_time = example["input_times"][0]
        # Round down to the nearest symbolic token step so segment offsets line up.
        start_time -= start_time % (1 / self.codec.steps_per_second)
        return {"est_tokens": tokens, "start_time": start_time, "raw_inputs": []}

    @staticmethod
    def _trim_eos(tokens, eos_id):
        tokens = np.array(tokens, np.int32)
        if eos_id in tokens:
            tokens = tokens[: np.argmax(tokens == eos_id)]
        return tokens
