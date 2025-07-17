import gradio as gr
import cv2
from detect import run_detection_and_log
from summary import generate_summary  # make sure summary.py exists

def process_video(video_file):
    input_path = video_file
    output_video = "output/hazard_output_logged.mp4"
    log_csv = "output/detections.csv"
    PREVIEW_IMAGE = "preview.jpg"

    # Generate preview frame from middle of video
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    ret, frame = cap.read()
    cap.release()

    if ret:
        cv2.imwrite(PREVIEW_IMAGE, frame)
    else:
        return "Error reading video", None, None, None

    # Show progress message
    progress_msg = "Preview ready. Running full detection..."

    # Run detection
    run_detection_and_log(input_path, output_video, log_csv)

    # Generate summary report
    summary_text = generate_summary(log_csv)

    return  progress_msg, PREVIEW_IMAGE, output_video, log_csv, summary_text


with gr.Blocks() as demo:
    gr.Markdown("## 🚲 Road Hazard Detection for Cyclists")
    gr.Markdown("Upload a road video. The model will detect hazards, log results, and summarize findings.")

    with gr.Row():
        video_input = gr.Video(label="Upload Road Video")

    run_btn = gr.Button("Run Detection")

    progress = gr.Textbox(label="Progress", interactive=False)
    preview = gr.Image(label="Detection Preview Frame")

    with gr.Row():
        output_video = gr.Video(label="Output Video with Detections")
        csv_file = gr.File(label="Detection Log (CSV)")
    
    summary_output = gr.Textbox(label="Hazard Summary", lines=10)

    run_btn.click(fn=process_video, inputs=video_input, outputs=[progress, preview, output_video, csv_file, summary_output])

demo.launch()
