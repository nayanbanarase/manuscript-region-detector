import argparse
import json
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = [
    "header",
    "footer",
    "main_text",
    "side_text",
    "filler",
]


def clamp_box(x1, y1, x2, y2, width, height):
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width - 1))
    y2 = max(0, min(int(y2), height - 1))
    return [x1, y1, x2, y2]


def add_detection(detections, label, confidence, box):
    detections.append(
        {
            "label": label,
            "confidence": round(float(confidence), 3),
            "bbox": box,
        }
    )


def detect_layout(image):
    """
    Baseline manuscript layout detector.

    This is a classical computer-vision baseline rather than
    a trained deep-learning detector. It estimates manuscript
    regions using page geometry and dark-pixel distributions.
    """

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Improve contrast and reduce small noise.
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold dark manuscript content.
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Remove very small isolated noise.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    detections = []

    # ---------------------------------------------------------
    # 1. Header
    # ---------------------------------------------------------
    header_height = max(1, int(height * 0.16))
    header_region = binary[:header_height, :]

    header_pixels = cv2.countNonZero(header_region)
    header_ratio = header_pixels / float(header_region.size)

    if header_ratio > 0.01:
        add_detection(
            detections,
            "header",
            min(0.95, 0.55 + header_ratio),
            clamp_box(
                width * 0.08,
                height * 0.03,
                width * 0.92,
                height * 0.16,
                width,
                height,
            ),
        )

    # ---------------------------------------------------------
    # 2. Footer
    # ---------------------------------------------------------
    footer_start = int(height * 0.84)
    footer_region = binary[footer_start:, :]

    footer_pixels = cv2.countNonZero(footer_region)
    footer_ratio = footer_pixels / float(footer_region.size)

    if footer_ratio > 0.01:
        add_detection(
            detections,
            "footer",
            min(0.95, 0.55 + footer_ratio),
            clamp_box(
                width * 0.08,
                height * 0.84,
                width * 0.92,
                height * 0.97,
                width,
                height,
            ),
        )

    # ---------------------------------------------------------
    # 3. Main text region
    # ---------------------------------------------------------
    middle_start = int(height * 0.16)
    middle_end = int(height * 0.84)

    middle_binary = binary[middle_start:middle_end, :]

    # Calculate amount of dark content in each vertical strip.
    column_density = np.sum(middle_binary > 0, axis=0)

    # Columns containing meaningful text.
    threshold = max(2, int(0.05 * (middle_end - middle_start)))
    active_columns = column_density > threshold

    if np.any(active_columns):
        active_indices = np.where(active_columns)[0]
        x_left = int(active_indices.min())
        x_right = int(active_indices.max())

        # Leave a small margin.
        x_left = max(0, x_left - int(width * 0.03))
        x_right = min(width - 1, x_right + int(width * 0.03))

        add_detection(
            detections,
            "main_text",
            0.72,
            clamp_box(
                x_left,
                middle_start,
                x_right,
                middle_end,
                width,
                height,
            ),
        )

    # ---------------------------------------------------------
    # 4. Side text
    # ---------------------------------------------------------
    # Look for content near the left/right page margins.
    side_width = int(width * 0.20)

    left_side = binary[middle_start:middle_end, :side_width]
    right_side = binary[middle_start:middle_end, width - side_width:]

    left_ratio = cv2.countNonZero(left_side) / float(left_side.size)
    right_ratio = cv2.countNonZero(right_side) / float(right_side.size)

    if left_ratio > 0.08:
        add_detection(
            detections,
            "side_text",
            min(0.90, 0.55 + left_ratio),
            clamp_box(
                0,
                middle_start,
                side_width,
                middle_end,
                width,
                height,
            ),
        )

    if right_ratio > 0.08:
        add_detection(
            detections,
            "side_text",
            min(0.90, 0.55 + right_ratio),
            clamp_box(
                width - side_width,
                middle_start,
                width,
                middle_end,
                width,
                height,
            ),
        )

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # 5. Filler
    # ---------------------------------------------------------
    # Filler represents page areas outside the primary text
    # regions, such as unused margins and blank page space.

    filler_boxes = [
    # Left margin
    (
        0,
        int(height * 0.16),
        int(width * 0.08),
        int(height * 0.84),
    ),

    # Right margin
    (
        int(width * 0.92),
        int(height * 0.16),
        width,
        int(height * 0.84),
    ),
]

    for x1, y1, x2, y2 in filler_boxes:
        add_detection(
            detections,
            "filler",
            0.60,
            clamp_box(
                x1,
                y1,
                x2,
                y2,
                width,
                height,
            ),
        )

    return detections

def draw_detections(image, detections):
    output = image.copy()

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        label = detection["label"]
        confidence = detection["confidence"]

        # Draw bounding box
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3,
        )

        # Label text
        text = f"{label} {confidence:.2f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness,
        )

        # Keep label inside the image
        label_y = max(text_height + 8, y1)

        # Label background
        cv2.rectangle(
            output,
            (x1, label_y - text_height - 8),
            (x1 + text_width + 10, label_y + baseline),
            (0, 255, 0),
            -1,
        )

        # Label text
        cv2.putText(
            output,
            text,
            (x1 + 5, label_y - 3),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    return output


def process_image(image_path, output_dir):
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read image: {image_path}")
        return

    height, width = image.shape[:2]

    detections = detect_layout(image)

    annotated = draw_detections(image, detections)

    output_dir.mkdir(parents=True, exist_ok=True)

    annotated_path = output_dir / f"{image_path.stem}_annotated.jpg"
    json_path = output_dir / f"{image_path.stem}.json"

    cv2.imwrite(str(annotated_path), annotated)

    result = {
        "image": image_path.name,
        "image_width": width,
        "image_height": height,
        "detections": detections,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Processed: {image_path.name}")
    print(f"  Annotated image: {annotated_path}")
    print(f"  JSON: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Manuscript page layout region detector"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to an image or folder of images",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output folder",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if input_path.is_file():
        process_image(input_path, output_dir)

    elif input_path.is_dir():
        image_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
        }

        image_files = sorted(
            [
                p
                for p in input_path.iterdir()
                if p.suffix.lower() in image_extensions
            ]
        )

        if not image_files:
            print("No image files found.")
            return

        for image_path in image_files:
            process_image(image_path, output_dir)

    else:
        print(f"Input path does not exist: {input_path}")


if __name__ == "__main__":
    main()