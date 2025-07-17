import pandas as pd
import os

def generate_summary(csv_path):
    if not os.path.exists(csv_path):
        return f"❌ Error: CSV file '{csv_path}' does not exist."

    try:
        df = pd.read_csv(csv_path)

        if df.empty:
            return "⚠️ No detections were logged. The video may contain no visible hazards."

        summary = []
        summary.append("📊 Hazard Detection Summary")
        summary.append("-" * 35)

        # Count by label
        label_counts = df['label'].value_counts()
        for label, count in label_counts.items():
            summary.append(f"{label}: {count} detection(s)")

        summary.append("\n🕒 Frame Range")
        summary.append(f"First Frame: {df['frame'].min()}")
        summary.append(f"Last Frame: {df['frame'].max()}")

        summary.append("\n📌 Confidence Range")
        summary.append(f"Min Confidence: {df['confidence'].min():.2f}")
        summary.append(f"Max Confidence: {df['confidence'].max():.2f}")

        return "\n".join(summary)

    except Exception as e:
        return f"❌ Failed to generate summary: {str(e)}"
