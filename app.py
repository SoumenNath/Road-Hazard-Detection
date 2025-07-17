import gradio as gr
import os
from detect import run_detection_and_log
from summary import generate_summary  # make sure summary.py exists

def process_video(video_file):
    input_path = video_file
    output_video = "output/hazard_output_logged.mp4"
    log_csv = "output/detections.csv"

    # Run detection
    run_detection_and_log(input_path, output_video, log_csv)

    # Generate summary report
    summary_text = generate_summary(log_csv)

    return output_video, log_csv, summary_text


with gr.Blocks() as demo:
    gr.Markdown("## 🚲 Road Hazard Detection for Cyclists")
    gr.Markdown("Upload a road video. The model will detect hazards, log results, and summarize findings.")

    with gr.Row():
        video_input = gr.Video(label="Upload Road Video")

    run_btn = gr.Button("Run Detection")

    with gr.Row():
        output_video = gr.Video(label="Output Video with Detections")
        csv_file = gr.File(label="Detection Log (CSV)")
    
    summary_output = gr.Textbox(label="Hazard Summary", lines=10)

    run_btn.click(fn=process_video, inputs=video_input, outputs=[output_video, csv_file, summary_output])

demo.launch()
