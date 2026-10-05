# Palm Vein Biometrics with Temperature Scanner

A Raspberry Pi kiosk application built for contact-tracing during the
COVID-19 period. It identifies a returning person by their palm-vein
pattern (via a CNN classifier) instead of touch-based fingerprint or
RFID, and logs a temperature reading alongside each verification.

Originally built as a college thesis project (2022). This repo contains
a cleaned-up version of the code plus the original as it was submitted -
see [Evolution](#evolution--lessons-learned) below.

## Features

- **Registration flow** - captures 100 images each of a person's left
  and right palm via a USB/Pi camera, alongside their personal details.
- **Preprocessing pipeline** - ROI crop, denoising, CLAHE contrast
  equalization, and morphological skeletonization to extract a
  vein-pattern image suitable for classification.
- **Verification flow** - captures a palm image and a contactless
  temperature reading (MLX90614 IR sensor), classifies the palm against
  registered users with a VGG16 transfer-learning model, and logs the
  result to MySQL.
- **Model training script** - trains the VGG16-based classifier on the
  preprocessed dataset and reports precision/recall/F1, a confusion
  matrix, and FAR/FRR/EER.

## Tech Stack

Python - Tkinter - OpenCV - TensorFlow/Keras (VGG16 transfer learning)
- MySQL - Raspberry Pi (Blinka/CircuitPython for I2C sensor access)

## Hardware Requirements

The GUI app (`src/palm_vein_gui.py`) targets a Raspberry Pi with:

- A USB or Pi camera (accessed via `cv2.VideoCapture(0)`)
- An MLX90614 contactless IR temperature sensor over I2C
- A MySQL server reachable on the network

`src/preprocess_dataset.py` and `src/train_cnn_model.py` have no
hardware dependency and run on any machine with the packages in
`requirements.txt` installed.

## Project Structure

```
.
├── src/
│   ├── palm_vein_gui.py        # Raspberry Pi kiosk app (registration + verification)
│   ├── preprocess_dataset.py   # Batch preprocessing for building a training set
│   └── train_cnn_model.py      # VGG16 transfer-learning training script
├── db/
│   └── schema.sql              # MySQL schema (reconstructed from the queries in the code)
├── docs/
│   └── original-2022-version/  # The code as originally submitted for the thesis
└── requirements.txt
```

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   On the Raspberry Pi itself, also install the CircuitPython/Blinka
   packages needed for `board` / `busio` / `adafruit_mlx90614` - see
   [Adafruit's Blinka setup guide](https://learn.adafruit.com/circuitpython-on-raspberrypi-linux)
   for your specific Pi model.

2. Create the database:
   ```
   mysql -u root -p < db/schema.sql
   ```

3. Set the required environment variables (see `src/palm_vein_gui.py`
   for the full list - DB host, user, password, database name, and the
   filesystem paths used for captured/processed images and the trained
   model file).

4. Run the app on the Pi:
   ```
   python3 src/palm_vein_gui.py
   ```

## How It Works

1. **Register** - collects a person's details, writes them to the
   `persons` table, then captures 100 images per hand.
2. **Preprocess** - each captured image is cropped to the palm region,
   denoised, contrast-equalized, and reduced to a skeletonized vein
   pattern; these become the training images for that user.
3. **Train** (`train_cnn_model.py`) - a VGG16 backbone (frozen early
   layers) is fine-tuned to classify users from their preprocessed palm
   images.
4. **Verify** - a fresh palm capture goes through the same
   preprocessing pipeline, gets classified, and - above a confidence
   threshold - is matched to a registered person; the temperature
   reading is logged against that match.

## Evolution / Lessons Learned

The version in `docs/original-2022-version/` is the code as it was
during the thesis - kept here on purpose rather than deleted. Revisiting
it later surfaced several real issues that the cleaned-up `src/` version
fixes:

- SQL queries built with raw string concatenation (a SQL-injection
  shaped hole) instead of parameterized queries.
- Four separate, nearly-identical blocks of code for rebuilding the
  home screen, and huge widget-parameter lists threaded through every
  function, instead of a single reusable navigation pattern.
- A `cv2.cvtColor(..., COLOR_BGR2RGB)` call on an image that's already
  single-channel grayscale by that point in the pipeline - which throws
  in a standard OpenCV build.
- A database connection that was only ever closed on the error path,
  never on success.

Comparing the two versions is a decent illustration of what a few years
and more practice will do to the same underlying idea.

## Limitations & Future Improvements

- No automated tests.
- Camera ROI coordinates are hardcoded to a specific camera position/
  resolution used during the thesis - would need recalibrating for any
  other physical setup.
- DB credentials/paths are currently read as plain constants at the top
  of the script - fine for a kiosk on a closed network, but should move
  to environment variables for anything more exposed.
- The classifier was trained on a small number of registered users;
  accuracy at larger scale is untested.

## License

MIT - see [LICENSE](LICENSE). This was originally a college thesis
project; feel free to reference it, but if you're a student working on
something similar, use it to learn from rather than to submit as your
own.
