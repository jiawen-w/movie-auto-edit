"""
movie-auto-edit: 电影自动剪辑工具
流程：场景分割 → 缩略图 → 人脸聚类/识别 → 筛选片段 → 按节拍合成 + 配乐
"""
import argparse
import ffmpy
import face_recognition
import subprocess as sp
import os
import shutil
import numpy as np
import json
import random


# ---------- Step 1: 场景分割 ----------

def split_video(movie_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cmd = (
        f'scenedetect --input "{movie_path}" '
        f'detect-content split-video --output "{output_dir}"'
    )
    print(f"[分割] {cmd}")
    p = sp.Popen(cmd, shell=True)
    p.wait()
    if p.returncode != 0:
        raise RuntimeError("场景分割失败，请确认 scenedetect 已正确安装")
    clips = [f for f in os.listdir(output_dir) if f.lower().endswith('.mp4')]
    print(f"[分割完成] 共 {len(clips)} 个片段 → {output_dir}")


# ---------- Step 2: 生成缩略图 ----------

def extract_frame(video_path, output_path, timestamp="00:00:02"):
    ff = ffmpy.FFmpeg(
        inputs={video_path: None},
        outputs={output_path: ['-ss', timestamp, '-vframes', '1', '-y']}
    )
    try:
        ff.run(stderr_nullfd=True)
        return True
    except Exception as e:
        print(f"  [警告] 提取帧失败 {os.path.basename(video_path)}: {e}")
        return False

def generate_thumbnails(clips_dir, thumbnail_dir):
    os.makedirs(thumbnail_dir, exist_ok=True)
    clips = [f for f in os.listdir(clips_dir) if f.lower().endswith('.mp4')]
    for clip in clips:
        thumb = os.path.splitext(clip)[0] + ".jpg"
        extract_frame(
            os.path.join(clips_dir, clip),
            os.path.join(thumbnail_dir, thumb)
        )
    print(f"[缩略图] 已生成 {len(clips)} 张 → {thumbnail_dir}")


# ---------- Step 3A: 自动聚类主角 ----------

def cluster_main_characters(thumbnail_dir, top_n=2):
    from sklearn.cluster import DBSCAN

    thumbs = [f for f in os.listdir(thumbnail_dir)
              if f.lower().endswith(('.jpg', '.png'))]
    print(f"[人脸聚类] 分析 {len(thumbs)} 张缩略图...")

    all_encodings, all_names = [], []
    for thumb in thumbs:
        try:
            img = face_recognition.load_image_file(
                os.path.join(thumbnail_dir, thumb))
            for enc in face_recognition.face_encodings(img):
                all_encodings.append(enc)
                all_names.append(thumb)
        except Exception as e:
            print(f"  [跳过] {thumb}: {e}")

    if len(all_encodings) < 2:
        raise ValueError("检测到的人脸太少（<2），无法自动聚类，请通过 --reference 提供参考图片")

    labels = DBSCAN(metric='euclidean', eps=0.6, min_samples=2).fit_predict(
        np.array(all_encodings))

    valid = labels[labels >= 0]
    if len(valid) == 0:
        raise ValueError("聚类失败，请降低 --tolerance 或通过 --reference 提供参考图片")

    unique, counts = np.unique(valid, return_counts=True)
    top_labels = unique[np.argsort(-counts)[:top_n]]

    encodings_arr = np.array(all_encodings)
    target_encodings = []
    for i, label in enumerate(top_labels):
        cluster = encodings_arr[labels == label]
        representative = np.mean(cluster, axis=0)
        target_encodings.append(representative)
        print(f"  主角 {i+1}: 出现在 {counts[np.where(unique==label)[0][0]]} 个片段中")

    return target_encodings


# ---------- Step 3B: 使用参考图片 ----------

def load_reference_encoding(image_path):
    img = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(img)
    if not encodings:
        raise ValueError(f"在参考图片中未检测到人脸: {image_path}")
    print(f"[参考人脸] 已加载: {image_path}")
    return [encodings[0]]


# ---------- Step 4: 筛选含主角片段 ----------

def filter_clips_by_face(clips_dir, thumbnail_dir, target_encodings,
                          output_dir, tolerance=0.55):
    os.makedirs(output_dir, exist_ok=True)
    thumbs = [f for f in os.listdir(thumbnail_dir)
              if f.lower().endswith(('.jpg', '.png'))]
    matched = []

    print(f"[筛选] 对 {len(thumbs)} 张缩略图进行人脸匹配...")
    for thumb in thumbs:
        try:
            img = face_recognition.load_image_file(
                os.path.join(thumbnail_dir, thumb))
            encodings = face_recognition.face_encodings(img)
            hit = any(
                np.min(face_recognition.face_distance(target_encodings, enc)) < tolerance
                for enc in encodings
            )
            if hit:
                clip_name = os.path.splitext(thumb)[0] + ".mp4"
                src = os.path.join(clips_dir, clip_name)
                if os.path.exists(src):
                    dest = os.path.join(output_dir, clip_name)
                    shutil.copy2(src, dest)
                    matched.append(dest)
                    print(f"  [匹配] {clip_name}")
        except Exception as e:
            print(f"  [跳过] {thumb}: {e}")

    print(f"[筛选完成] {len(matched)} / {len(thumbs)} 个片段包含主角")
    return matched


# ---------- Step 5: 合成剪辑 + 配乐 ----------

def get_duration(path):
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
           '-of', 'json', path]
    out = sp.run(cmd, capture_output=True, text=True)
    return float(json.loads(out.stdout)['format']['duration'])

def get_beat_times(music_path):
    try:
        import librosa
        y, sr = librosa.load(music_path)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        times = librosa.frames_to_time(beats, sr=sr).tolist()
        dur = librosa.get_duration(y=y, sr=sr)
        print(f"[音乐] BPM={tempo:.1f}, 节拍数={len(times)}, 时长={dur:.1f}s")
        return times, dur
    except ImportError:
        print("[音乐] librosa 未安装，使用均匀分割（pip install librosa 可启用节拍对齐）")
        return None, get_duration(music_path)

def assemble_edit(clips, music_path, output_path, temp_dir):
    os.makedirs(temp_dir, exist_ok=True)

    beat_times, music_dur = get_beat_times(music_path)
    print(f"[合成] 音乐时长 {music_dur:.1f}s，从 {len(clips)} 个片段中选取素材")

    if beat_times:
        step = max(1, round(len(beat_times) / (music_dur / 3.0)))
        cuts = [0.0] + [beat_times[i]
                        for i in range(step, len(beat_times), step)
                        if beat_times[i] < music_dur]
    else:
        n = max(1, int(music_dur / 3.5))
        cuts = [i * music_dur / n for i in range(n)]
    cuts.append(music_dur)
    seg_durs = [cuts[i+1] - cuts[i] for i in range(len(cuts)-1)]

    clip_durs = {}
    for c in clips:
        try:
            clip_durs[c] = get_duration(c)
        except:
            clip_durs[c] = 4.0

    pool = clips.copy()
    random.shuffle(pool)
    while len(pool) < len(seg_durs):
        extra = clips.copy()
        random.shuffle(extra)
        pool.extend(extra)

    trimmed = []
    for i, (clip, seg_dur) in enumerate(zip(pool, seg_durs)):
        out = os.path.join(temp_dir, f"seg_{i:04d}.mp4")
        clip_dur = clip_durs.get(clip, 4.0)
        max_start = max(0.0, clip_dur - seg_dur - 0.3)
        start = random.uniform(0, max_start)

        ff = ffmpy.FFmpeg(
            inputs={clip: f'-ss {start:.3f}'},
            outputs={out: [
                '-t', f'{seg_dur:.3f}',
                '-vf', ('scale=1920:1080:force_original_aspect_ratio=decrease,'
                        'pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black'),
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-an', '-y'
            ]}
        )
        try:
            ff.run(stderr_nullfd=True)
            trimmed.append(out)
        except Exception as e:
            print(f"  [警告] 裁剪第 {i} 段失败: {e}")

    if not trimmed:
        raise RuntimeError("所有片段裁剪均失败")

    concat_txt = os.path.join(temp_dir, "concat.txt")
    with open(concat_txt, 'w', encoding='utf-8') as f:
        for t in trimmed:
            f.write(f"file '{t}'\n")

    silent_video = os.path.join(temp_dir, "silent.mp4")
    print("[合成] 拼接片段...")
    sp.run(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_txt,
            '-c', 'copy', '-y', silent_video], check=True)

    print("[合成] 混入音乐...")
    ff_final = ffmpy.FFmpeg(
        inputs={silent_video: None, music_path: None},
        outputs={output_path: [
            '-map', '0:v', '-map', '1:a',
            '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
            '-shortest', '-y'
        ]}
    )
    ff_final.run()
    print(f"[完成] 输出 → {output_path}")


# ---------- 主流程 ----------

def main():
    parser = argparse.ArgumentParser(
        description="movie-auto-edit: 自动识别主角并生成电影混剪（配乐版）"
    )
    parser.add_argument("movie",   help="输入电影文件路径")
    parser.add_argument("music",   help="背景音乐文件路径")
    parser.add_argument("output",  help="输出视频文件路径")
    parser.add_argument(
        "--reference", "-r",
        default=None,
        help="主角参考图片路径（留空则自动识别出镜最多的主角）"
    )
    parser.add_argument(
        "--top", "-n",
        type=int, default=2,
        help="自动模式下识别几位主角（默认 2）"
    )
    parser.add_argument(
        "--tolerance", "-t",
        type=float, default=0.55,
        help="人脸匹配容差 0.4~0.65，越小越严格（默认 0.55）"
    )
    args = parser.parse_args()

    movie_path  = args.movie
    music_path  = args.music
    output_path = args.output

    base = os.path.join(
        os.path.dirname(os.path.abspath(movie_path)),
        "_edit_" + os.path.splitext(os.path.basename(movie_path))[0]
    )
    dirs = {
        'clips':      os.path.join(base, "01_clips"),
        'thumbnails': os.path.join(base, "02_thumbnails"),
        'matched':    os.path.join(base, "03_matched"),
        'temp':       os.path.join(base, "04_temp"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    print("\n===== Step 1: 场景分割 =====")
    split_video(movie_path, dirs['clips'])

    print("\n===== Step 2: 生成缩略图 =====")
    generate_thumbnails(dirs['clips'], dirs['thumbnails'])

    print("\n===== Step 3: 人脸识别 =====")
    if args.reference:
        target_encodings = load_reference_encoding(args.reference)
    else:
        target_encodings = cluster_main_characters(
            dirs['thumbnails'], top_n=args.top)

    print("\n===== Step 4: 筛选主角片段 =====")
    matched = filter_clips_by_face(
        dirs['clips'], dirs['thumbnails'],
        target_encodings, dirs['matched'],
        tolerance=args.tolerance
    )
    if not matched:
        raise ValueError(
            "未找到包含目标人物的片段，请调整 --tolerance 或通过 --reference 提供参考图片")

    print("\n===== Step 5: 合成剪辑 =====")
    assemble_edit(matched, music_path, output_path, dirs['temp'])

    print(f"\n===== 完成！输出: {output_path} =====")


if __name__ == '__main__':
    main()
