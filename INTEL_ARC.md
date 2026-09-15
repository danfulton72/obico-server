# Intel Arc / OpenVINO support for Obico

This branch adds Intel Arc GPU acceleration to Obico's ML service by running the ONNX model through ONNX Runtime's OpenVINO Execution Provider.

It has been tested on an **Intel Arc A310** on Debian 12 / OpenMediaVault, with the GPU exposed to Docker through `/dev/dri`.

## What this branch changes

The Arc-specific implementation is intentionally kept small:

- `docker-compose.arc.yml` overrides only the `ml_api` service.
- `ml_api/Dockerfile.arc` uses the OpenVINO Ubuntu runtime image.
- `ml_api/lib/onnx.py` selects `OpenVINOExecutionProvider` for GPU inference.
- `ml_api/lib/detection_model.py` prefers the ONNX model rather than CUDA/Darknet.
- `ml_api/verify_arc.py` verifies OpenVINO device discovery and model compilation.

The normal Obico web, task, Redis, database, media and printer-client behaviour is unchanged.

## Tested software combination

The working image currently uses:

```text
OpenVINO runtime image: openvino/ubuntu22_runtime:2025.1.0
onnxruntime-openvino:    1.23.0
OpenVINO device:        GPU
```

Those versions are paired deliberately. If either package is changed in future, verify that ONNX Runtime can still load the OpenVINO Execution Provider and compile the Obico model.

## Host requirements

The Intel GPU must be visible on the host:

```bash
ls -l /dev/dri
```

A working Arc setup should normally expose something similar to:

```text
/dev/dri/card0
/dev/dri/renderD128
```

You can also confirm the kernel driver with:

```bash
lspci -nnk | grep -A3 -Ei 'VGA|Display'
```

The tested Arc A310 uses the Linux `i915` driver.

## Verify OpenVINO can see the GPU

Before building Obico, this standalone container test is useful:

```bash
docker run --rm \
  --device=/dev/dri:/dev/dri \
  --group-add="$(stat -c '%g' /dev/dri/renderD128)" \
  openvino/ubuntu22_runtime:2025.4.1 \
  python3 -c 'from openvino import Core; c=Core(); print(c.available_devices); [print(d, "=", c.get_property(d, "FULL_DEVICE_NAME")) for d in c.available_devices]'
```

Expected output should include both CPU and GPU, for example:

```text
['CPU', 'GPU']
CPU = Intel(R) Xeon(R) CPU ...
GPU = Intel(R) Arc(TM) A310 ...
```

Host-side `clinfo` is not required for this setup; the important path is OpenVINO inside the container with access to `/dev/dri`.

## Environment variables

Add the render-device group ID and OpenVINO device to `.env`:

```bash
echo "RENDER_GID=$(stat -c '%g' /dev/dri/renderD128)" >> .env
echo "OPENVINO_DEVICE=GPU" >> .env
```

Do not commit `.env` because it may contain secrets such as `DJANGO_SECRET_KEY`, SMTP credentials or API keys.

The Arc override consumes these values as follows:

```yaml
services:
  ml_api:
    environment:
      OPENVINO_DEVICE: '${OPENVINO_DEVICE-GPU}'

    devices:
      - /dev/dri:/dev/dri

    group_add:
      - '${RENDER_GID}'
```

## Build

From the Obico repository root:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.arc.yml \
  build ml_api
```

The Arc image downloads Obico's ONNX failure-detection model into:

```text
/model_cache/ml_api/onnx/model-weights.onnx
```

## Verify the model before starting the stack

Run the included verification script:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.arc.yml \
  run --rm ml_api \
  python3 /app/verify_arc.py
```

Successful output should show:

```text
GPU = Intel(R) Arc(TM) ...
Available providers: ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
Requested OpenVINO device: GPU
Active providers: ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
Arc/OpenVINO Obico model compilation: OK
```

`CPUExecutionProvider` remains registered as a fallback. The important part is that `OpenVINOExecutionProvider` is available, requested first, and successfully compiles the model for `GPU`.

## Start Obico

Use both Compose files whenever starting, stopping, rebuilding or inspecting this deployment:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.arc.yml \
  up -d
```

Check status:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.arc.yml \
  ps
```

Follow ML logs:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.arc.yml \
  logs -f ml_api
```

A successful startup should contain lines similar to:

```text
Trying to load weights: /model_cache/ml_api/onnx/model-weights.onnx - use_gpu = True
ONNX Runtime available providers: ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
ONNX Runtime requested providers: [('OpenVINOExecutionProvider', {'device_type': 'GPU'}), 'CPUExecutionProvider']
ONNX Runtime active providers: ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
Succeeded!
```

## Confirm the running container still sees Arc

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.arc.yml \
  exec ml_api \
  python3 -c 'from openvino import Core; c=Core(); print([(d,c.get_property(d,"FULL_DEVICE_NAME")) for d in c.available_devices])'
```

The output should include the Intel Arc GPU.

## Ports

The ML API listens on port `3333` **inside the Docker network only**. It is intentionally not published to the host.

Therefore this is expected to fail from the host:

```bash
curl http://127.0.0.1:3333/hc/
```

Use the container/network health checks instead:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.arc.yml \
  exec ml_api \
  wget -qO- http://127.0.0.1:3333/hc/
```

or from the web container:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.arc.yml \
  exec web \
  wget -qO- http://ml_api:3333/hc/
```

The Obico web UI remains published on port `3334` unless changed separately.

## Updating from upstream Obico

This fork keeps upstream Obico as a separate remote and carries the Arc changes on the `intel-arc-openvino` branch.

A typical update flow is:

```bash
git fetch upstream
git switch intel-arc-openvino
git rebase upstream/release
```

Resolve any conflicts in the Arc-specific files, then rebuild and verify the ML image:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.arc.yml \
  build ml_api


docker compose \
  -f docker-compose.yml \
  -f docker-compose.arc.yml \
  run --rm ml_api \
  python3 /app/verify_arc.py


docker compose \
  -f docker-compose.yml \
  -f docker-compose.arc.yml \
  up -d
```

After a successful rebase, update the fork with:

```bash
git push --force-with-lease origin intel-arc-openvino
```

## Troubleshooting

### `GPU` is missing from OpenVINO devices

Check that `/dev/dri/renderD128` exists and that the container receives both the device and its render-group GID.

```bash
stat -c '%g' /dev/dri/renderD128
```

Make sure that value matches `RENDER_GID` in `.env`.

### `OpenVINOExecutionProvider` is unavailable

Check the installed package inside the ML image:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.arc.yml \
  run --rm ml_api \
  python3 -c 'import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())'
```

The output must include:

```text
OpenVINOExecutionProvider
```

### OpenVINO provider is present but model compilation fails

Re-run:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.arc.yml \
  run --rm ml_api \
  python3 /app/verify_arc.py
```

If this started after changing OpenVINO or ONNX Runtime versions, return to the tested pairing listed above before troubleshooting further.

## Known-good hardware

Confirmed working in this fork:

```text
Intel Arc A310
Linux DRM device: /dev/dri/renderD128
OpenVINO target: GPU
```

Other Intel Arc / Xe GPUs should use the same OpenVINO device path in principle, but are not yet documented here as tested.