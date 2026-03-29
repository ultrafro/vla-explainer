"""
Generate narration for tactile sensing explainer and mux with video.
"""
import asyncio
import edge_tts
import subprocess
import os
import json

VOICE = "en-US-GuyNeural"
RATE = "-5%"

SCENES = [
    "S01_Title", "S02_WhyTouch", "S03_Sensors", "S04_3DViTac",
    "S05_3DViTacArch", "S06_FusionDetail", "S07_Sparsh", "S08_TANDEM",
    "S09_TacSL", "S10_FusionTypes", "S11_TactileInVLA", "S12_HandObject",
    "S13_Limitations", "S14_Frontier", "S15_Summary",
]

SCRIPTS = {
    "S01_Title": (
        "Tactile sensing for robot learning. When you pick up a glass, you don't just see it — "
        "you feel its weight, its temperature, whether it's slipping. This sense of touch is what "
        "separates clumsy robots from dexterous ones. In this video, we'll explore how the latest "
        "research fuses tactile sensing with vision and language models, creating robots that can "
        "truly feel their way through complex manipulation tasks."
    ),
    "S02_WhyTouch": (
        "Why does touch matter? Vision alone has fundamental blind spots. When a robot grasps an "
        "object, its fingers occlude the camera view. It can't sense contact forces, can't detect "
        "when an object starts slipping, and can't feel whether something is fragile or rigid. "
        "Touch provides exactly what vision lacks: contact geometry, normal and shear forces, "
        "real-time slip detection, and material recognition. As researchers like to say, "
        "vision tells you what to grasp, but touch tells you how to grasp."
    ),
    "S03_Sensors": (
        "There are four main families of tactile sensors used in robotics research. "
        "Optical sensors like GelSight and DIGIT use a camera behind a soft gel membrane. When "
        "something presses against the gel, the deformation is captured as an RGB image, giving "
        "incredibly high-resolution contact geometry at around 640 by 480 pixels. "
        "Piezoelectric arrays, like those used in 3D-ViTac, are pressure-sensitive grids with "
        "16 by 16 sensing units per finger pad, giving 256 taxels at about 32 hertz, at a cost "
        "of under 20 dollars per pad. "
        "Capacitive sensors like RoboSkin provide flexible coverage over large areas. "
        "And BioTac from SynTouch is the gold standard — a multimodal sensor with force, "
        "vibration, and temperature, closest to a human fingertip, but expensive."
    ),
    "S04_3DViTac": (
        "Let's dive into 3D-ViTac, a breakthrough paper from Columbia and UIUC presented at "
        "CoRL 2024. The key innovation is fusing vision and touch in 3D space. "
        "The hardware uses custom piezoelectric sensor pads — four per hand in a bimanual setup — "
        "with 16 by 16 sensing units per pad, giving 1024 total tactile sensing units across "
        "both hands. Each pad costs less than 20 dollars, making this accessible for real labs. "
        "The critical insight is that both visual and tactile data can be represented as 3D "
        "point clouds, allowing them to be merged into a single unified spatial representation."
    ),
    "S05_3DViTacArch": (
        "Here's the 3D-ViTac architecture in detail. The pipeline has three stages. "
        "First, an RGB-D camera captures the scene and generates a raw 3D point cloud. This is "
        "downsampled using Farthest Point Sampling to N-vis equals 512 points. "
        "Second, the tactile sensor readings are converted to 3D coordinates using forward "
        "kinematics — the robot knows where its fingertips are in space, so each tactile reading "
        "gets a 3D position plus a pressure channel, giving N-tac equals 256 points with 4 dimensions. "
        "Third, these two point clouds are concatenated into a unified 3D visuo-tactile "
        "representation of 768 points, which is fed into a PointNet++ backbone. "
        "The output conditions a diffusion policy that generates 7 degree-of-freedom actions."
    ),
    "S06_FusionDetail": (
        "Let's break down exactly how the fusion works. On the visual side, the RGB-D camera "
        "produces a raw point cloud that gets downsampled to 512 points in R-3 space. "
        "On the tactile side, each sensor pad's 256 taxel readings get transformed into 3D "
        "world coordinates using the robot's forward kinematics, with an extra pressure channel, "
        "giving a tensor in R-256-by-4. "
        "The magic happens at the merge step: because both representations live in the same "
        "3D coordinate frame, they can simply be concatenated — 512 visual points plus 256 "
        "tactile points equals 768 points total. "
        "This combined point cloud goes through PointNet++, which processes 3D point sets "
        "with hierarchical feature learning. The resulting features condition a denoising "
        "diffusion policy that outputs smooth, multimodal action sequences."
    ),
    "S07_Sparsh": (
        "Sparsh, from Meta's FAIR lab, takes a completely different approach. Instead of "
        "building task-specific tactile models, Sparsh creates a foundation model for touch. "
        "The idea is borrowed from computer vision: use self-supervised learning to pre-train "
        "on massive amounts of unlabeled tactile data. "
        "Sparsh was trained on over 460,000 tactile images from multiple sensor types — "
        "GelSight, DIGIT, and GelSight Mini — using three SSL methods: Masked Autoencoders, "
        "DINO, and JEPA. The resulting representations generalize across different sensors "
        "and tasks without needing task-specific labels. "
        "They also introduce TacBench, a standardized benchmark for evaluating tactile "
        "representations across six tasks including force estimation, slip detection, "
        "and grasp stability prediction. Sparsh with DINO and Sparsh with JEPA emerged as "
        "the top performing models."
    ),
    "S08_TANDEM": (
        "TANDEM, from Shuran Song's lab at Columbia, addresses a different question: "
        "where should a robot touch to get the most useful information? "
        "This is the active tactile exploration problem. The architecture has two co-trained "
        "components. The Explorer generates a policy for where and how to touch objects. "
        "The Discriminator takes the tactile data collected by the Explorer and tries to "
        "identify what object was touched. "
        "These two modules are trained together: the Explorer learns to provide maximally "
        "informative touch sequences, while the Discriminator learns to classify from those "
        "sequences. The result significantly outperforms random exploration strategies "
        "and is robust to sensor noise — the robot intelligently decides where to touch."
    ),
    "S09_TacSL": (
        "One of the biggest bottlenecks in tactile robot learning is data collection. "
        "Real tactile data requires a physical robot making real contacts, which is slow "
        "and expensive. TacSL, from NVIDIA and the University of Washington, solves this "
        "with GPU-accelerated tactile simulation. "
        "TacSL is a library that plugs into Isaac Sim and simulates visuotactile sensors "
        "using finite element methods on the GPU. It generates both RGB tactile images "
        "and force fields at 350 times faster than prior simulation methods. "
        "It also introduces ACOCD, a novel reinforcement learning algorithm that uses "
        "active critic distillation to efficiently learn tactile-based policies in simulation "
        "that can transfer to the real world."
    ),
    "S10_FusionTypes": (
        "Now let's zoom out and look at the four main architectures for fusing tactile "
        "and visual data. "
        "First, early concatenation: the simplest approach. Encode each modality separately, "
        "concatenate the feature vectors, and pass through an MLP. Simple but loses "
        "fine-grained interactions between modalities. "
        "Second, cross-attention fusion: visual features serve as queries, tactile features "
        "as keys and values. The attention mechanism learns which visual regions should attend "
        "to which tactile signals. More expressive but more compute. "
        "Third, 3D spatial fusion, as in 3D-ViTac: project both modalities into a shared 3D "
        "coordinate space and process with point cloud networks. Preserves geometric structure. "
        "Fourth, token interleaving, the VLA-native approach: convert all modalities — "
        "images, tactile, text — into tokens and feed them as one sequence into a transformer. "
        "This is the direction the field is moving, as it leverages the same architecture "
        "that powers large language models."
    ),
    "S11_TactileInVLA": (
        "Here's how tactile data would enter a full VLA pipeline. "
        "Three inputs arrive: a camera image, tactile sensor readings, and a language instruction. "
        "Each goes through its own encoder. The ViT encoder produces 256 visual tokens of "
        "dimension 768. A tactile encoder — which could be a Sparsh-pretrained model, a CNN, "
        "or a point cloud network — produces 64 tactile tokens of the same dimension. "
        "The language tokenizer produces 32 text tokens. "
        "All 352 tokens are concatenated into a single sequence and fed through the "
        "transformer backbone. Here's the key: through self-attention, every tactile token "
        "can attend to every visual and language token. The model learns when and where "
        "touch information matters — for example, attending strongly to tactile signals "
        "during contact but ignoring them during approach. "
        "The fused output goes through an action head that produces 7-DoF robot commands."
    ),
    "S12_HandObject": (
        "This paper from MIT takes tactile sensing in a different direction — modeling "
        "full hand-object dynamics. Using a high-resolution tactile glove, they capture "
        "the complete interaction between a human hand and objects during everyday tasks "
        "like juggling, stick balancing, and throwing. "
        "A predictive model and contrastive learning framework processes these tactile "
        "sequences and remarkably can reconstruct the 3D trajectories of both the hand "
        "and the object from touch signals alone. "
        "This opens the door to imitation learning from human demonstrations, where a robot "
        "learns manipulation skills not just from watching, but from feeling how a human "
        "expert handles objects."
    ),
    "S13_Limitations": (
        "Despite the rapid progress, tactile sensing for robotics faces significant challenges. "
        "Sensor fragility: optical sensors like GelSight degrade with use — gel membranes tear "
        "and markers wear off, requiring frequent replacement. "
        "There's no standard interface: every sensor type outputs different data in different "
        "formats, unlike the relatively standardized world of cameras. "
        "Coverage is sparse: most sensors only cover fingertips, missing palm, wrist, and arm contacts. "
        "The sim-to-real gap persists: simulated tactile signals don't perfectly match real "
        "sensor noise and drift. "
        "Data scarcity is a major issue: there are only hundreds of thousands of tactile images "
        "available, compared to billions for vision. "
        "And there's a latency versus resolution tradeoff: high-resolution sensors like GelSight "
        "run at only 30 hertz, while faster sensors give lower resolution."
    ),
    "S14_Frontier": (
        "Research is advancing on multiple fronts. "
        "Foundation models for touch, like Sparsh, show that self-supervised pre-training "
        "can create general-purpose tactile representations that work across sensor types. "
        "Tactile simulation at scale, through TacSL and Isaac Sim, enables millions of "
        "simulated tactile interactions for training without real robots. "
        "Whole-body tactile skins aim to cover entire robot surfaces, not just fingertips, "
        "enabling richer contact awareness. "
        "Tactile-VLA co-training is the next frontier: training VLAs with touch data from "
        "the start, rather than adding it as an afterthought. "
        "Cross-modal supervision uses vision labels to train tactile-only models, enabling "
        "zero-shot transfer. "
        "And dexterous in-hand manipulation combines multi-finger control with dense tactile "
        "feedback to achieve human-level object reorientation."
    ),
    "S15_Summary": (
        "Let's recap. Touch gives robots what vision can't: contact forces, slip detection, "
        "and geometry under occlusion. "
        "3D-ViTac showed us how to fuse vision and touch in a shared 3D space using point clouds "
        "and diffusion policies. "
        "Sparsh proved that self-supervised foundation models work for touch, generalizing across "
        "sensor types. "
        "TANDEM taught robots to actively decide where to touch for maximum information. "
        "TacSL brought GPU-accelerated simulation to tactile learning, making data collection scalable. "
        "The field is converging on a unified approach: tokenize everything — images, touch, language — "
        "and let a transformer learn the cross-modal relationships. "
        "We're still early. Data, sensors, and standards are the bottleneck. But the pieces are "
        "coming together fast. Thanks for watching."
    ),
}


def get_duration(path):
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(json.loads(result.stdout)["format"]["duration"])
    except (json.JSONDecodeError, KeyError):
        return 0.0


async def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    audio_dir = "audio_tactile"
    os.makedirs(audio_dir, exist_ok=True)

    print("Generating narration...")
    for scene in SCENES:
        audio_path = os.path.join(audio_dir, f"{scene}.mp3")
        text = SCRIPTS[scene]
        communicate = edge_tts.Communicate(text, VOICE, rate=RATE)
        await communicate.save(audio_path)
        print(f"  {scene}: {audio_path}")

    video_dir = os.path.join("media", "videos", "tactile_explainer", "1080p60")
    muxed_dir = "muxed_tactile"
    os.makedirs(muxed_dir, exist_ok=True)

    muxed_files = []
    for scene in SCENES:
        video_path = os.path.join(video_dir, f"{scene}.mp4")
        audio_path = os.path.join(audio_dir, f"{scene}.mp3")
        muxed_path = os.path.join(muxed_dir, f"{scene}.mp4")

        if not os.path.exists(video_path):
            print(f"  SKIP: {video_path}")
            continue

        vid_dur = get_duration(video_path)
        aud_dur = get_duration(audio_path)
        print(f"  {scene}: video={vid_dur:.1f}s, audio={aud_dur:.1f}s")

        if aud_dur > vid_dur and vid_dur > 0:
            speed = vid_dur / aud_dur
            cmd = [
                "ffmpeg", "-y", "-i", video_path, "-i", audio_path,
                "-filter_complex", f"[0:v]setpts={1/speed}*PTS[v]",
                "-map", "[v]", "-map", "1:a",
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-c:a", "aac", "-b:a", "192k", "-shortest",
                muxed_path,
            ]
        else:
            cmd = [
                "ffmpeg", "-y", "-i", video_path, "-i", audio_path,
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
                muxed_path,
            ]
        subprocess.run(cmd, capture_output=True)
        muxed_files.append(muxed_path)

    concat_file = "concat_tactile.txt"
    with open(concat_file, "w") as f:
        for mf in muxed_files:
            f.write(f"file '{mf}'\n")

    output = "Tactile_Sensing_Explainer.mp4"
    print(f"\nConcatenating {len(muxed_files)} scenes...")
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k", output,
    ]
    subprocess.run(cmd, capture_output=True)
    os.remove(concat_file)

    size_mb = os.path.getsize(output) / (1024 * 1024)
    print(f"\nDone! {os.path.abspath(output)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    asyncio.run(main())
