import pandas as pd
from collections import Counter

# === Configuration ===
CSV_PATH = "output/detections.csv"
VIDEO_FPS = 30  # change this to match your video's FPS
SUMMARY_LIMIT = 10  # top-N hazard classes to show

def generate_summary():
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"❌ File not found: {CSV_PATH}")
        return

    if df.empty:
        print("⚠️ The CSV log is empty. No detections to summarize.")
        return

    print("📊 Generating hazard summary...\n")

    # Total frames processed
    total_frames = df["frame"].nunique()
    duration_sec = total_frames / VIDEO_FPS
    duration_min = duration_sec / 60

    # Count hazards per label
    label_counts = Counter(df["label"])

    print(f"🎞️ Total frames analyzed: {total_frames}")
    print(f"⏱️ Estimated duration: {duration_sec:.2f} seconds ({duration_min:.2f} minutes)")
    print("\n📌 Hazard Detections by Class:")

    print(f"{'Hazard Class':<20} {'Count':<10} {'/min':<10} {'/sec':<10}")
    print("-" * 50)
    for label, count in label_counts.most_common(SUMMARY_LIMIT):
        per_min = count / duration_min if duration_min > 0 else 0
        per_sec = count / duration_sec if duration_sec > 0 else 0
        print(f"{label:<20} {count:<10} {per_min:<10.2f} {per_sec:<10.2f}")

    print("\n✅ Summary complete.")

if __name__ == "__main__":
    generate_summary()
