# Digit Recognizer

Local recognition of handwritten digits with a Swing/FlatLaf interface and Brain4J inference.

![Java](https://img.shields.io/badge/Java-22-ED8B00)
![Gradle](https://img.shields.io/badge/Gradle-9.8-02303A)

## What it does

- Vector 28×28 canvas, rendered at screen resolution while drawing.
- Preprocessing: crop the ink, scale to 20×20, and center on the center of mass.
- Local CNN: `Conv 32 → MaxPool → Conv 64 → MaxPool → Dense 128 → Dense 10`.
- Typographic panel with a segmented confidence meter and ten animated probability rows.
- Asynchronous inference with a 120 ms debounce, without blocking the interface.
- `Use GPU` checkbox: moves the model to the GPU and uses hardware acceleration, with automatic CPU fallback when the backend is unavailable.
- Model saved and loaded with Brain4J's native `BrainFormat` (ZIP archive with `config.json` and `weights.safetensors`).

Included model accuracy: **94.8%** on the MNIST test set (10 epochs, lr 1e-4).

## Requirements

- JDK 22 (the Foojay Gradle resolver downloads the toolchain if missing).
- Internet connection on first run for dependencies and the MNIST dataset.
- GPU inference requires the Slang compiler (`slangc`) in `PATH` and the NVIDIA NVRTC libraries (`libnvrtc`); without them the app stays on the CPU.

## Run

```bash
./gradlew run
```

Shortcuts:

- `C` — clears the canvas
- `Ctrl+Z` — undoes the last stroke

## Training

```bash
./gradlew train --args="--epochs 10"
```

The model is written to `src/main/resources/models/mnist-cnn.zip`.

Useful options:

```bash
./gradlew train --args="--epochs 10 --batch 128 --lr 0.0001"
./gradlew train --args="--limit 3000 --epochs 5"
```

## Test

```bash
./gradlew test
```

## Structure

```text
src/main/java/it/echo/digitrecognizer/
├── DigitRecognizerApp.java     # entry point
├── MainFrame.java              # window and wiring
├── ai/
│   ├── DigitRecognizer.java    # loading + inference
│   ├── MnistCnn.java           # CNN architecture
│   └── MnistTrainer.java       # training CLI
├── core/
│   └── DigitPreprocessor.java  # input normalization
└── ui/
    ├── DigitCanvas.java
    ├── PredictionPanel.java
    ├── ProbabilityPanel.java
    ├── InputPreview.java
    ├── Theme.java
    └── Card.java
```

## Brain4J note

The project uses the native `BrainFormat` and the `3.0.0-5fac901a-SNAPSHOT` snapshot, which correctly initializes the layer, activation, weight initializer, and clipper registries. The `.zip` file is the native container produced by `ModelIO.save`: it is not a generic archive, but the Brain4J format with `config.json` and `weights.safetensors`.

## GPU note

The `Use GPU` checkbox creates a Silicon/CUDA device (or Metal on macOS) and moves the model parameters to the GPU. The first use compiles the kernels through Slang and can take a few seconds. If `slangc` is not installed, the app shows `GPU unavailable: Slang not found` and continues using the CPU. You can provide an explicit path with the `SLANGC` environment variable or the `-Dsilicon.slangc=/path/to/slangc` property.

On Linux the CUDA backend also requires NVRTC. If `slangc` reports `failed to load downstream compiler 'nvrtc'`, install the NVRTC libraries; on Fedora 44:

```bash
sudo dnf install cuda-nvrtc-13-2 cuda-nvrtc-devel-13-2
```

The `cuda-nvrtc-devel` package provides the `libnvrtc.so` symlink used by the Slang loader.
