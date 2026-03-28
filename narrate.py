"""
Generate narration audio for each scene, then mux with video.
Uses Microsoft Edge TTS (free, high quality).
"""
import asyncio
import edge_tts
import subprocess
import os
import json

VOICE = "en-US-GuyNeural"  # Clear, 3B1B-ish male narrator voice
RATE = "-5%"  # Slightly slower for clarity

SCRIPTS = {
    "Scene01_Title": (
        "Vision Language Action models, or VLAs, represent a breakthrough in robotics. "
        "They combine three capabilities into a single neural network: vision, to see the world "
        "through cameras; language, to understand human instructions; and action, to directly "
        "control a robot's movements. The key idea is elegant: pixels go in, and robot actions "
        "come out, all from one unified model."
    ),
    "Scene02_WhyVLAs": (
        "To understand why VLAs matter, consider the traditional robotics pipeline. "
        "Historically, robots used a chain of separate modules: perception to process sensor data, "
        "state estimation to build a world model, planning to decide what to do, and control to "
        "execute movements. Each module was hand-engineered, and errors cascaded between stages. "
        "The system was brittle and couldn't leverage the vast knowledge available on the internet. "
        "VLAs replace this entire pipeline with end-to-end learning: give the model an image and "
        "a language instruction, and it directly outputs the robot action."
    ),
    "Scene03_Architecture": (
        "Let's break down the VLA architecture into its four core components. "
        "First, a vision encoder, typically a Vision Transformer like SigLIP or DINOv2, "
        "converts the camera image into a sequence of visual tokens. "
        "Second, a language encoder tokenizes the human instruction and passes it through "
        "a large language model backbone to produce contextual embeddings. "
        "Third, a transformer backbone fuses these visual and language tokens together using "
        "cross-attention, creating a rich multimodal representation. "
        "Finally, an action head decodes this fused representation into concrete robot actions, "
        "like changes in position, rotation, and gripper state."
    ),
    "Scene04_VisionEncoder": (
        "The vision encoder is how the robot sees. It takes a raw camera image and divides it "
        "into a grid of patches, typically 16 by 16 pixels each. Each patch is linearly projected "
        "into an embedding vector, creating a sequence of visual tokens. "
        "These tokens are then processed by a Vision Transformer, which uses self-attention to "
        "capture spatial relationships across the image. "
        "Most VLAs use pretrained encoders like SigLIP, DINOv2, or CLIP, which have already "
        "learned rich visual representations from billions of image-text pairs on the internet. "
        "These encoders can be frozen or fine-tuned during robot training."
    ),
    "Scene05_Language": (
        "Language conditioning is what makes VLAs so flexible. Instead of programming specific "
        "behaviors, you simply tell the robot what to do in natural language. "
        "The instruction, like 'pick up the red cup and place it on the plate', is first "
        "tokenized into subword tokens. These tokens are fed through a large language model "
        "backbone, such as Llama or PaLM, which produces contextual embeddings that capture "
        "the semantic meaning of the task. "
        "This is the key insight: the LLM provides deep understanding of what the task means, "
        "enabling the robot to generalize to new instructions it hasn't seen before."
    ),
    "Scene06_ActionHead": (
        "The action head is where understanding becomes movement. There are two main approaches. "
        "The first is continuous regression, where a multilayer perceptron outputs the mean and "
        "variance of a Gaussian distribution for each action dimension. This is used in models "
        "like RT-2 and Octo. "
        "The second approach is discrete tokenization, where continuous action values are binned "
        "into discrete tokens, typically 256 bins per dimension. The model then autoregressively "
        "decodes these tokens, just like generating text. This is used in RT-2, Pi Zero, and OpenVLA. "
        "In both cases, the output is a 7 degree-of-freedom action: three for position, "
        "three for rotation, and one for the gripper."
    ),
    "Scene07_Training": (
        "Training a VLA happens in two stages. "
        "Stage one is vision-language pre-training. You start with a pretrained vision-language "
        "model that has been trained on billions of image-text pairs from the internet. "
        "This gives the model a strong foundation in visual understanding and language reasoning. "
        "Stage two is robot action fine-tuning. You add an action head to the model and train it "
        "on robot demonstration datasets. Each training example is a tuple of an image, a language "
        "instruction, and the corresponding robot action. "
        "Key datasets include Open X-Embodiment with over one million episodes, Bridge V2, "
        "DROID, and the RT datasets from Google. "
        "The training objective is straightforward: next-token prediction on action tokens, "
        "using cross-entropy loss."
    ),
    "Scene08_Inference": (
        "During inference, the VLA runs in a tight control loop on the robot. "
        "Step one: capture an image from the robot's camera. "
        "Step two: tokenize both the image and the language instruction. "
        "Step three: run a forward pass through the transformer to get action predictions. "
        "Step four: decode the predicted action and send it to the robot's motors. "
        "Then the loop repeats with the next camera frame. "
        "This typically runs at 5 to 10 hertz, with about 100 to 200 milliseconds of latency per step. "
        "A technique called action chunking helps: instead of predicting one action at a time, "
        "the model predicts several steps ahead, which smooths out the motion and reduces "
        "the impact of inference latency."
    ),
    "Scene09_KeyModels": (
        "Let's look at the landmark VLA models. "
        "RT-2, from Google DeepMind in 2023, was a pioneer, building on PaLM-E with 55 billion "
        "parameters and co-fine-tuning on robot data. "
        "Octo, from UC Berkeley in 2024, introduced a diffusion-based action head and supports "
        "multiple robot embodiments. It's fully open source. "
        "OpenVLA, from Stanford and Toyota Research, built on Llama 2 with 7 billion parameters "
        "and a SigLIP encoder. Also open source and designed to be fine-tunable. "
        "Pi Zero, from Physical Intelligence, uses flow matching for action generation with "
        "3 billion parameters. "
        "And Gemini Robotics from Google DeepMind in 2025, built on the Gemini 2.0 backbone, "
        "pushing toward dexterous manipulation."
    ),
    "Scene10_Limitations": (
        "Despite the progress, VLAs face significant limitations. "
        "First, they're data hungry. Collecting real robot demonstrations is expensive and slow, "
        "unlike the nearly unlimited text and image data available online. "
        "Second, there's a generalization gap. Models struggle with objects, environments, and "
        "tasks not seen during training. "
        "Third, latency. Large models with billions of parameters are slow for real-time control, "
        "where every millisecond matters. "
        "Fourth, safety. There are no built-in guarantees that the robot won't take dangerous actions. "
        "Fifth, the sim-to-real gap. Training in simulation doesn't transfer perfectly to the real world. "
        "And sixth, long-horizon tasks. Current models struggle with complex multi-step tasks "
        "that require planning over many minutes."
    ),
    "Scene11_Research": (
        "Research is advancing rapidly on multiple fronts. "
        "Flow matching, as used in Pi Zero, replaces discrete action tokens with continuous "
        "flow-based generation, enabling smoother and more precise actions. "
        "Diffusion policies model the full distribution of possible actions using denoising "
        "diffusion, handling multimodal action distributions naturally. "
        "Researchers are studying scaling laws for robotics: does more data and bigger models "
        "reliably lead to better robots? Early evidence says yes. "
        "Sim-to-real transfer is improving through domain randomization and adaptation techniques. "
        "Multi-embodiment models aim to train one model that controls many different robots, "
        "as in the Open X-Embodiment initiative. "
        "And world models learn the physics and dynamics of the environment, allowing robots "
        "to plan in imagination before acting in the real world."
    ),
    "Scene12_Summary": (
        "Let's recap what we've learned. "
        "VLAs unify vision, language, and action into a single end-to-end model. "
        "They're built on foundation model backbones: large language models combined with "
        "pretrained vision encoders. "
        "They're trained on large-scale robot demonstration data in a two-stage process. "
        "They enable language-conditioned robotic manipulation, where you can tell a robot "
        "what to do in plain English. "
        "They're still limited by data requirements, latency, and generalization challenges. "
        "But progress is rapid, with new architectures appearing every few months. "
        "The next generation of robots may truly be programmed with words, not code. "
        "Thanks for watching."
    ),
}

async def generate_audio(scene_name, text, output_path):
    communicate = edge_tts.Communicate(text, VOICE, rate=RATE)
    await communicate.save(output_path)
    print(f"  Generated: {output_path}")

async def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    audio_dir = "audio"
    os.makedirs(audio_dir, exist_ok=True)

    # Generate all audio files
    print("Generating narration audio...")
    for scene_name, script in SCRIPTS.items():
        audio_path = os.path.join(audio_dir, f"{scene_name}.mp3")
        await generate_audio(scene_name, script, audio_path)

    # Now mux each scene's video with its audio, adjusting to match
    video_dir = os.path.join("media", "videos", "vla_explainer", "1080p60")
    muxed_dir = "muxed"
    os.makedirs(muxed_dir, exist_ok=True)

    scene_names = list(SCRIPTS.keys())
    muxed_files = []

    for scene_name in scene_names:
        video_path = os.path.join(video_dir, f"{scene_name}.mp4")
        audio_path = os.path.join(audio_dir, f"{scene_name}.mp3")
        muxed_path = os.path.join(muxed_dir, f"{scene_name}.mp4")

        if not os.path.exists(video_path):
            print(f"  WARNING: {video_path} not found, skipping")
            continue

        # Get durations
        vid_dur = get_duration(video_path)
        aud_dur = get_duration(audio_path)
        print(f"  {scene_name}: video={vid_dur:.1f}s, audio={aud_dur:.1f}s")

        # Strategy: use the longer of the two as the target duration
        # If audio is longer, slow down video (speed factor < 1)
        # If video is longer, pad audio with silence at the end
        if aud_dur > vid_dur and vid_dur > 0:
            # Slow down video to match audio length
            speed = vid_dur / aud_dur
            # Use setpts to slow video, atempo for audio passthrough
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", audio_path,
                "-filter_complex",
                f"[0:v]setpts={1/speed}*PTS[v]",
                "-map", "[v]", "-map", "1:a",
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                muxed_path,
            ]
        else:
            # Video is longer or equal - just add audio, pad audio with silence
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", audio_path,
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                muxed_path,
            ]

        subprocess.run(cmd, capture_output=True)
        muxed_files.append(muxed_path)

    # Concatenate all muxed files
    concat_file = "concat_muxed.txt"
    with open(concat_file, "w") as f:
        for mf in muxed_files:
            f.write(f"file '{mf}'\n")

    output = "VLA_Explainer_Full.mp4"
    print(f"\nConcatenating {len(muxed_files)} narrated scenes...")

    # Re-encode to ensure consistent format for concat
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_file,
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        output,
    ]
    subprocess.run(cmd, capture_output=True)
    os.remove(concat_file)

    size_mb = os.path.getsize(output) / (1024 * 1024)
    print(f"\nDone! Output: {os.path.abspath(output)} ({size_mb:.1f} MB)")


def get_duration(path):
    """Get media duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except (json.JSONDecodeError, KeyError):
        return 0.0


if __name__ == "__main__":
    asyncio.run(main())
