# Running on an NVIDIA GPU

Pronunciada runs on the CPU by default.

On NVIDIA GPUs, pass `device="cuda"` to `Scorer` (or `--device cuda` to `check`,
`score`, or `serve`). Use an environment with the GPU build of ONNX Runtime and
matching CUDA and cuDNN libraries. `precision="fp16"` (CLI: `--precision fp16`)
selects the existing half-precision export. The device setting applies to the
acoustic model. See the [ONNX Runtime CUDA setup guide](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

To run this checkout with CUDA on Windows, create a separate Python environment
and use the GPU build of ONNX Runtime as the sole ONNX Runtime distribution:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install "onnxruntime-gpu[cuda,cudnn]" "numpy>=1.24" "kaldi-native-fbank>=1.18" "soundfile>=0.12"
.\.venv\Scripts\python.exe -m pronunciada serve --device cuda --precision fp16
```
