"""
Render all VLA explainer scenes and concatenate into one video.
Usage: python render.py
"""
import subprocess
import sys
import os

SCENES = [
    "Scene01_Title",
    "Scene02_WhyVLAs",
    "Scene03_Architecture",
    "Scene04_VisionEncoder",
    "Scene05_Language",
    "Scene06_ActionHead",
    "Scene07_Training",
    "Scene08_Inference",
    "Scene09_KeyModels",
    "Scene10_Limitations",
    "Scene11_Research",
    "Scene12_Summary",
]

QUALITY = "h"  # h = 1080p, m = 720p, l = 480p

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Render each scene
    for scene in SCENES:
        print(f"\n{'='*60}")
        print(f"  Rendering: {scene}")
        print(f"{'='*60}")
        cmd = [
            sys.executable, "-m", "manim",
            "-q" + QUALITY,
            "vla_explainer.py",
            scene,
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"ERROR rendering {scene}")
            sys.exit(1)

    # Find all rendered mp4s
    media_dir = os.path.join("media", "videos", "vla_explainer", "1080p60")
    if not os.path.isdir(media_dir):
        # Try alternate path
        media_dir = os.path.join("media", "videos", "vla_explainer", "1080p30")

    scene_files = []
    for scene in SCENES:
        mp4 = os.path.join(media_dir, f"{scene}.mp4")
        if os.path.exists(mp4):
            scene_files.append(mp4)
        else:
            print(f"WARNING: {mp4} not found, skipping")

    if not scene_files:
        print("No scene files found!")
        sys.exit(1)

    # Write concat list for ffmpeg
    concat_file = "concat_list.txt"
    with open(concat_file, "w") as f:
        for sf in scene_files:
            f.write(f"file '{sf}'\n")

    # Concatenate with ffmpeg
    output = "VLA_Explainer_Full.mp4"
    print(f"\nConcatenating {len(scene_files)} scenes into {output}...")
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_file,
        "-c", "copy",
        output,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("ffmpeg concat failed, trying re-encode method...")
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_file,
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            output,
        ]
        subprocess.run(cmd)

    print(f"\nDone! Output: {os.path.abspath(output)}")
    # Clean up
    os.remove(concat_file)

if __name__ == "__main__":
    main()
