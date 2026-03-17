#!/usr/bin/env python3
"""
Extract hair colors from an image using Mediapipe multiclass segmentation.

Approach:
1. Mediapipe selfie_multiclass model → direct hair mask (label=1)
2. K-means clustering on hair pixels → find dominant dark/light colors
3. Return the darkest cluster as root color, lightest as tip color
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.cluster import KMeans
import sys
import json
import os

VisionRunningMode = mp.tasks.vision.RunningMode

# Segmentation labels: 0=background, 1=hair, 2=body-skin, 3=face-skin, 4=clothes, 5=others
HAIR_LABEL = 1


def _remove_ambient_cast(rgb: np.ndarray) -> np.ndarray:
    """
    Remove ambient light color cast from a hair color.

    Black/dark hair has almost zero real saturation — it just reflects ambient
    light (often blue or purple from sky/room). This corrects by reducing
    saturation proportionally to how dark the color is:
      - brightness=0   → saturation scaled to 0   (pure black)
      - brightness=80  → saturation scaled to 0.2 (very dark, near-achromatic)
      - brightness=160 → saturation scaled to 0.65 (medium — some color preserved)
      - brightness=220 → saturation unchanged      (light beige/blonde fully kept)

    Also shifts hue slightly toward warm (orange/brown) range for dark hair,
    since cool-cast dark hair is almost never correct for rendering.
    """
    pixel = rgb.reshape(1, 1, 3).astype(np.uint8)
    hsv = cv2.cvtColor(pixel, cv2.COLOR_RGB2HSV).reshape(3).astype(float)
    h, s, v = hsv  # H: 0-179, S: 0-255, V: 0-255

    brightness = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]

    # Saturation scale: dark → less saturation (removes ambient cast)
    # Smooth ramp: 0 at brightness=0, 1 at brightness=220+
    sat_scale = np.clip(brightness / 220.0, 0.0, 1.0) ** 0.6
    s_new = s * sat_scale

    # For dark colors with a cool (blue/purple) hue cast: nudge toward warm brown
    # Blue/purple range in OpenCV HSV: H ~ 100-160
    if brightness < 120 and 90 <= h <= 160:
        # Shift hue toward warm brown (H~15 in OpenCV = ~30° which is brown/orange)
        warmth_shift = (160 - h) / 160.0 * 20  # max 20 units toward warm
        h = max(0, h - warmth_shift)

    hsv_new = np.array([h, s_new, v], dtype=np.float32).reshape(1, 1, 3).astype(np.uint8)
    rgb_new = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2RGB).reshape(3)
    return rgb_new


class HairColorExtractor:
    def __init__(self, face_landmarker_path, segmenter_path=None):
        # Face Landmarker (for fallback / forehead reference)
        base_options = python.BaseOptions(
            model_asset_path=face_landmarker_path,
            delegate=mp.tasks.BaseOptions.Delegate.CPU
        )
        options = vision.FaceLandmarkerOptions(
            running_mode=VisionRunningMode.IMAGE,
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=10,
            min_face_detection_confidence=0.1,
            min_face_presence_confidence=0.1,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        # Image Segmenter (multiclass: background/hair/body-skin/face-skin/clothes/others)
        if segmenter_path is None:
            segmenter_path = os.path.join(
                os.path.dirname(face_landmarker_path), "selfie_multiclass.tflite"
            )
        seg_base = python.BaseOptions(
            model_asset_path=segmenter_path,
            delegate=mp.tasks.BaseOptions.Delegate.CPU
        )
        seg_options = vision.ImageSegmenterOptions(
            base_options=seg_base,
            running_mode=VisionRunningMode.IMAGE,
            output_confidence_masks=False,
            output_category_mask=True,
        )
        self.segmenter = vision.ImageSegmenter.create_from_options(seg_options)

    def extract_hair_colors(self, image_path, n_clusters=5, debug_path=None):
        """
        Extract dark (root) and light (tip) hair colors from image.

        Steps:
        1. Run multiclass segmentation → hair mask (label=1)
        2. K-means cluster the hair pixels
        3. Pick darkest and lightest significant clusters
        """
        img = cv2.imread(os.path.expanduser(image_path))
        if img is None:
            print(f"ERROR: Could not load image: {image_path}")
            return None, None

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        print(f"Image size: {w}x{h}")

        # Run segmentation
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        seg_result = self.segmenter.segment(mp_img)
        cat_mask = seg_result.category_mask.numpy_view().squeeze()  # (H, W)

        # Hair mask
        hair_mask = (cat_mask == HAIR_LABEL).astype(np.uint8)
        hair_pixel_count = hair_mask.sum()
        total_pixels = h * w
        hair_pct = hair_pixel_count / total_pixels * 100
        print(f"Hair pixels: {hair_pixel_count} ({hair_pct:.1f}% of image)")

        if hair_pixel_count < 100:
            print("WARNING: Very few hair pixels (bald/buzzcut?)")
            print("  Using face-skin region darkened as fallback")
            # Fallback: detect face, sample above forehead
            det_result = self.detector.detect(mp_img)
            if len(det_result.face_landmarks) > 0:
                lm = det_result.face_landmarks[0]
                forehead = (int(lm[10].x * w), int(lm[10].y * h))
                y_start = max(0, forehead[1] - 60)
                y_end = max(0, forehead[1] - 5)
                x_start = max(0, forehead[0] - 80)
                x_end = min(w, forehead[0] + 80)
                region = img_rgb[y_start:y_end, x_start:x_end]
                if region.size > 0:
                    color = np.median(region, axis=(0, 1)).astype(np.uint8)
                    dark_hex = '#{:02x}{:02x}{:02x}'.format(*color)
                    lighter = np.clip(color.astype(float) * 1.3, 0, 255).astype(np.uint8)
                    light_hex = '#{:02x}{:02x}{:02x}'.format(*lighter)
                    print(f"\n✓ Extracted colors (fallback):")
                    print(f"  Dark (root):  {dark_hex} RGB({color[0]}, {color[1]}, {color[2]})")
                    print(f"  Light (tip):  {light_hex} RGB({lighter[0]}, {lighter[1]}, {lighter[2]})")
                    return dark_hex, light_hex
            return None, None

        # Extract hair pixels
        hair_pixels = img_rgb[hair_mask == 1]
        print(f"Total hair pixels for clustering: {len(hair_pixels)}")

        # Save debug image if requested
        if debug_path:
            debug_img = img_rgb.copy()
            # Dim non-hair, keep hair bright
            debug_img[hair_mask == 0] = (debug_img[hair_mask == 0] * 0.3).astype(np.uint8)
            # Draw hair outline
            contours, _ = cv2.findContours(hair_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 2)
            cv2.imwrite(debug_path, cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
            print(f"  Debug image saved: {debug_path}")

        # Subsample for K-means speed
        max_samples = 15000
        if len(hair_pixels) > max_samples:
            indices = np.random.choice(len(hair_pixels), max_samples, replace=False)
            sample_pixels = hair_pixels[indices].astype(np.float32)
        else:
            sample_pixels = hair_pixels.astype(np.float32)

        # K-means clustering
        actual_clusters = min(n_clusters, len(sample_pixels) // 10)
        actual_clusters = max(2, actual_clusters)
        kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
        kmeans.fit(sample_pixels)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_

        # Get cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\nK-means clusters ({len(centers)}):")
        for i, (center, count) in enumerate(zip(centers, counts)):
            r, g, b = center.astype(int)
            brightness = 0.299 * r + 0.587 * g + 0.114 * b
            pct = count / len(labels) * 100
            hex_c = '#{:02x}{:02x}{:02x}'.format(int(r), int(g), int(b))
            print(f"  Cluster {i}: {hex_c} RGB({r},{g},{b}) brightness={brightness:.0f} ({pct:.1f}%)")

        # Compute brightness of each cluster center
        brightness = 0.299 * centers[:, 0] + 0.587 * centers[:, 1] + 0.114 * centers[:, 2]
        sorted_idx = np.argsort(brightness)

        # Dark = darkest significant cluster (at least 5% of hair pixels)
        dark_idx = None
        for idx in sorted_idx:
            if counts[idx] / len(labels) > 0.05:
                dark_idx = idx
                break
        if dark_idx is None:
            dark_idx = sorted_idx[0]

        # Light = lightest significant cluster (different from dark)
        light_idx = None
        for idx in reversed(sorted_idx):
            if counts[idx] / len(labels) > 0.05 and idx != dark_idx:
                light_idx = idx
                break
        if light_idx is None:
            light_idx = sorted_idx[-1]

        dark_rgb = np.clip(centers[dark_idx], 0, 255).astype(np.uint8)
        light_rgb = np.clip(centers[light_idx], 0, 255).astype(np.uint8)

        # If dark and light are very similar, use percentiles instead
        dark_b = 0.299 * dark_rgb[0] + 0.587 * dark_rgb[1] + 0.114 * dark_rgb[2]
        light_b = 0.299 * light_rgb[0] + 0.587 * light_rgb[1] + 0.114 * light_rgb[2]

        if abs(light_b - dark_b) < 15:
            print("\n  Clusters too similar, using percentile approach")
            pixel_b = 0.299 * hair_pixels[:, 0].astype(float) + \
                      0.587 * hair_pixels[:, 1].astype(float) + \
                      0.114 * hair_pixels[:, 2].astype(float)
            dark_mask_p = pixel_b <= np.percentile(pixel_b, 20)
            light_mask_p = pixel_b >= np.percentile(pixel_b, 80)
            if dark_mask_p.sum() > 0:
                dark_rgb = np.median(hair_pixels[dark_mask_p], axis=0).astype(np.uint8)
            if light_mask_p.sum() > 0:
                light_rgb = np.median(hair_pixels[light_mask_p], axis=0).astype(np.uint8)

        # Remove ambient light color cast.
        # Black/dark hair reflects ambient light (blue, purple) which biases the
        # extracted color. In HSV: darken the saturation proportionally to how dark
        # the color is, so very dark hair approaches a neutral dark brown/black.
        dark_rgb = _remove_ambient_cast(dark_rgb)
        light_rgb = _remove_ambient_cast(light_rgb)

        # If all clusters are still relatively bright (brightness > 80),
        # the hair is likely being lit heavily. Scale down the dark color
        # so it actually reads as dark in the render.
        all_brightness = [0.299*c[0] + 0.587*c[1] + 0.114*c[2] for c in centers]
        max_brightness = max(all_brightness)
        dark_b_final = 0.299 * dark_rgb[0] + 0.587 * dark_rgb[1] + 0.114 * dark_rgb[2]
        if dark_b_final > 80 and max_brightness < 160:
            # All clusters are mid-range — darken the root
            scale = 0.6
            dark_rgb = np.clip(dark_rgb.astype(float) * scale, 0, 255).astype(np.uint8)
            print(f"  Ambient brightness correction: darkened root by {scale:.1f}x")

        dark_hex = '#{:02x}{:02x}{:02x}'.format(dark_rgb[0], dark_rgb[1], dark_rgb[2])
        light_hex = '#{:02x}{:02x}{:02x}'.format(light_rgb[0], light_rgb[1], light_rgb[2])

        print(f"\n✓ Extracted colors:")
        print(f"  Dark (root):  {dark_hex} RGB({dark_rgb[0]}, {dark_rgb[1]}, {dark_rgb[2]})")
        print(f"  Light (tip):  {light_hex} RGB({light_rgb[0]}, {light_rgb[1]}, {light_rgb[2]})")

        return dark_hex, light_hex


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_hair_colors.py <image_path> [mediapipe_model_path] [--debug]")
        sys.exit(1)

    image_path = sys.argv[1]
    mediapipe_model_path = "./inference/assets/face_landmarker.task"
    for arg in sys.argv[2:]:
        if not arg.startswith('--'):
            mediapipe_model_path = arg
            break
    debug = '--debug' in sys.argv

    extractor = HairColorExtractor(mediapipe_model_path)

    debug_path = None
    if debug:
        base = os.path.splitext(os.path.basename(image_path))[0]
        debug_path = f"debug_hair_mask_{base}.png"

    dark_hex, light_hex = extractor.extract_hair_colors(image_path, debug_path=debug_path)

    if dark_hex and light_hex:
        result = {
            "dark_root_color": dark_hex,
            "light_tip_color": light_hex
        }
        print(f"\nResult JSON: {json.dumps(result)}")


if __name__ == '__main__':
    main()
