import gradio as gr
import os
import cv2
import subprocess
import sys

WORKING_DIR = os.getcwd()
DEFAULT_OUTPUT_DIR = os.path.join(WORKING_DIR, "outputs")

def get_video_info(video_path):
    if not video_path:
        return 0
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    max_frames = int(fps * duration)
    cap.release()
    return max_frames

def run_inference(video_path, ref_image_path, 
                 max_frame_num, guidance_scale, batch_frames, 
                 sample_stride, overlap, height, width):
    command = [
        sys.executable, "run_controlnext.py",
        "--pretrained_model_name_or_path", "stabilityai/stable-video-diffusion-img2vid-xt-1-1",
        "--controlnext_path", "pretrained/controlnet.bin",
        "--unet_path", "pretrained/unet.bin",
        "--output_dir", DEFAULT_OUTPUT_DIR,
        "--validation_control_video_path", video_path,
        "--ref_image_path", ref_image_path,
        "--max_frame_num", str(max_frame_num),
        "--guidance_scale", str(guidance_scale),
        "--batch_frames", str(batch_frames),
        "--sample_stride", str(sample_stride),
        "--overlap", str(overlap),
        "--height", str(height),
        "--width", str(width)
    ]
    
    try:
        # Create the output directory if it doesn't exist
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        
        # Execute the command and capture output
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        output_text = ""
        for line in process.stdout:
            output_text += line
            yield output_text
        
        process.wait()
        if process.returncode != 0:
            yield output_text + f"\nProcess ended with return code {process.returncode}"
    except Exception as e:
        yield f"Error occurred: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("# ControlNeXt-SVD v2 WebUI")
    
    with gr.Row():
        with gr.Column():
            ref_image = gr.Image(label="Reference Image", type="filepath")
        with gr.Column():
            video_path = gr.Video(label="Control Video")
    
    with gr.Row():
        gr.Textbox(
            label="Output Directory",
            value=DEFAULT_OUTPUT_DIR,
            interactive=False
        )

    with gr.Accordion("Advanced Settings", open=False):
        max_frames = gr.Number(value=0, label="Max Frame Number")
        guidance_scale = gr.Number(value=3, label="Guidance Scale")
        batch_frames = gr.Number(value=20, label="Batch Frames")
        sample_stride = gr.Number(value=2, label="Sample Stride")
        overlap = gr.Number(value=6, label="Overlap")
        height = gr.Number(value=768, label="Height")
        width = gr.Number(value=512, label="Width")
    
    with gr.Row():
        run_btn = gr.Button("Run Inference", size="large")
    
    output = gr.Textbox(label="Command Output", lines=10)
    
    with gr.Accordion("Model Paths", open=False):
        with gr.Row():
            with gr.Column():
                gr.Textbox(value="stabilityai/stable-video-diffusion-img2vid-xt-1-1", 
                           label="Model Path", interactive=False)
            with gr.Column():
                gr.Textbox(value="pretrained/controlnet.bin", 
                           label="ControlNext Path", interactive=False)
            with gr.Column():
                gr.Textbox(value="pretrained/unet.bin", 
                           label="UNet Path", interactive=False)
    
    video_path.change(fn=get_video_info, 
                      inputs=[video_path], 
                      outputs=[max_frames])
    
    run_btn.click(fn=run_inference, 
                  inputs=[video_path, ref_image, 
                          max_frames, guidance_scale, batch_frames,
                          sample_stride, overlap, height, width],
                  outputs=[output])

if __name__ == "__main__":
    demo.launch()