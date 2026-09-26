# Manuscript Specific Layout Region Detection

## Technical Assignment

This project implements a manuscript page layout region detection pipeline for identifying structural regions in historical manuscript images.

The system processes individual images or batches of images and produces:

- Annotated images with detected regions
- JSON metadata containing region labels, confidence scores, and bounding boxes

## Required Classes

The assignment defines five manuscript layout categories:

1. `header`
2. `footer`
3. `main_text`
4. `side_text`
5. `filler`

## Project Structure

```text
manuscript-region-detector/
├── data/
│   └── test_images/
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── data.yaml
├── models/
├── results/
├── src/
├── inference.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Approach

This project uses a computer-vision-based baseline to detect layout regions in manuscript images.

The image is processed using grayscale conversion, image smoothing, thresholding, and spatial analysis of detected text regions. Heuristic rules are then used to identify the different manuscript page regions.

The system supports all five required classes:

- `header`
- `footer`
- `main_text`
- `side_text`
- `filler`

The `filler` class is handled using unused page-margin areas identified during layout analysis.

This implementation is designed as a baseline solution and does not assume the availability of annotated training data.

## Input

The inference pipeline accepts either:

- A single manuscript image
- A folder containing multiple manuscript images

For batch inference, run:

```bash
python inference.py --input ./data/test_images --output ./results
```

## Output

For each input image, the system generates two output files.

### 1. Annotated image

Contains:

- Bounding boxes
- Region labels
- Confidence scores

### 2. JSON metadata

Contains:

- Image name
- Image dimensions
- Detected region labels
- Confidence scores
- Bounding-box coordinates

The generated files are saved in the `results/` directory.

Example output:

```text
results/
├── manuscript_sample_1_annotated.jpg
├── manuscript_sample_1.json
├── manuscript_sample_2_annotated.jpg
└── manuscript_sample_2.json
```

Example JSON structure:

```json
{
  "image": "manuscript_sample_1.png",
  "image_width": 2047,
  "image_height": 774,
  "detections": [
    {
      "label": "header",
      "confidence": 0.57,
      "bbox": [163, 23, 1883, 123]
    },
    {
      "label": "main_text",
      "confidence": 0.72,
      "bbox": [171, 123, 2046, 650]
    }
  ]
}
```

## Sample Images

The manuscript images supplied with the assignment are stored in:

```text
data/test_images/
```

These images are treated as sample/evaluation inputs.

They are not assumed to be training data because bounding-box annotations were not provided for the supplied images.

## Running the Project

### 1. Install dependencies

Create and activate a Python virtual environment if required, then install the project dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run inference

Run the following command from the project root:

```bash
python inference.py --input ./data/test_images --output ./results
```

### 3. Check the results

After inference completes, open the `results/` folder to view:

- Annotated manuscript images
- JSON metadata files

## Limitations

This implementation is a computer-vision-based baseline using heuristic layout analysis rather than a trained deep-learning detection model.

The detected regions and confidence scores are estimates. Results may vary for manuscripts with substantially different page layouts, text arrangements, writing styles, or decorative elements.

The approach can be improved using a sufficiently large, manually annotated manuscript dataset and a trained object-detection model.

## Reproducibility

Run the following command from the project root:

```bash
python inference.py --input ./data/test_images --output ./results
```

The same input images will be processed, and the resulting annotated images and JSON metadata will be written to the `results/` directory.

## Requirements

The main dependencies are listed in `requirements.txt`.

Install them using:

```bash
pip install -r requirements.txt
```

## Submission Contents

The project submission contains:

- Source code
- Command-line inference script
- Requirements file
- README documentation
- Sample input images
- Generated sample outputs
- Project configuration files

## Future Improvements

Possible improvements include:

- Creating a larger manually annotated manuscript dataset
- Training a dedicated object-detection model
- Improving region boundary detection
- Improving confidence estimation
- Handling more diverse manuscript layouts
- Evaluating the system using standard object-detection metrics