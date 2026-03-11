import json
import os
import re
from typing import List, Tuple

class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size=200, chunk_overlap=0, length_function=len, is_separator_regex=False, separators=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.is_separator_regex = is_separator_regex
        self.separators = separators or ["\n\n", "\n", " ", ""]
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks using recursive character splitting"""
        chunks = []
        self._split_text_recursive(text, 0, chunks)
        return chunks
    
    def _split_text_recursive(self, text: str, start_idx: int, chunks: List[str]):
        if self.length_function(text) <= self.chunk_size:
            chunks.append(text)
            return
        
        for separator in self.separators:
            if separator:
                splits = text.split(separator)
                current_chunk = ""
                for i, split_text in enumerate(splits):
                    if i > 0:
                        test_chunk = current_chunk + separator + split_text
                    else:
                        test_chunk = current_chunk + split_text
                    
                    if self.length_function(test_chunk) <= self.chunk_size:
                        current_chunk = test_chunk
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                            remaining = separator.join(splits[i:])
                            self._split_text_recursive(remaining, start_idx + len(chunks), chunks)
                            return
                        else:
                            break
                if current_chunk:
                    chunks.append(current_chunk)
                    return
            else:
                chunks.append(text[:self.chunk_size])
                if len(text) > self.chunk_size:
                    remaining = text[self.chunk_size - self.chunk_overlap:]
                    self._split_text_recursive(remaining, start_idx + len(chunks), chunks)
                return

def milliseconds_to_srt_time(ms: int) -> str:
    """Convert milliseconds to SRT time format (HH:MM:SS,mmm)"""
    hours = ms // 3600000
    minutes = (ms % 3600000) // 60000
    seconds = (ms % 60000) // 1000
    milliseconds = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

def get_timestamp_for_text_segment(text_segment: str, full_asr_text: str, timestamps: List[List[int]]) -> Tuple[int, int]:
    """Get start and end timestamp for a text segment"""
    def clean_text(text):
        return re.sub(r'[^\u4e00-\u9fff\w]', '', text)
    
    clean_full = clean_text(full_asr_text)
    clean_segment = clean_text(text_segment)

    start_pos = clean_full.find(clean_segment)
    if start_pos == -1:
        return 0, 0
    
    end_pos = start_pos + len(clean_segment)

    char_index = 0
    start_time = 0
    end_time = 0
    
    for i, (start_ms, end_ms) in enumerate(timestamps):
        if char_index == start_pos:
            start_time = start_ms
        if char_index == end_pos - 1:
            end_time = end_ms
            break
        char_index += 1

    if end_time == 0 and timestamps:
        end_time = timestamps[-1][1]

    if start_time == 0 and timestamps:
        start_time = timestamps[0][0]
    
    return start_time, end_time

def process_video_segments(input_json_path: str, output_dir: str):
    """Process long video ASR content and split into segments with timestamps, each video saved as separate JSON file"""

    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")

    print(f"正在读取JSON文件: {input_json_path}")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        video_data = json.load(f)
    
    print(f"总共需要处理 {len(video_data)} 个视频")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", "。", "！", "？", "，", ",", ".", "!", "?", " ", ""]
    )
    
    processed_count = 0
    
    for i, video_item in enumerate(video_data):
        video_id = video_item.get('id', f'video_{i}')
        asr_text = video_item.get('ASR_text', '')
        timestamps = video_item.get('ASR_timestamp', [])
        
        print(f"正在处理视频 {i+1}/{len(video_data)}: {video_id}")
        print(f"  原始文本长度: {len(asr_text)} 字符")
        print(f"  时间戳数量: {len(timestamps)} 个")
        
        if not asr_text or not timestamps:
            print(f"  ⚠️  视频 {video_id} 缺少ASR文本或时间戳，跳过处理")
            continue

        text_segments = text_splitter.split_text(asr_text)
        print(f"  分割为 {len(text_segments)} 个片段")

        video_segments = []
        for j, segment in enumerate(text_segments):
            if segment.strip():
                try:
                    start_ms, end_ms = get_timestamp_for_text_segment(segment, asr_text, timestamps)

                    start_time_str = milliseconds_to_srt_time(start_ms)
                    end_time_str = milliseconds_to_srt_time(end_ms)
                    
                    segment_info = {
                        "segment_id": f"{video_id}_{j+1}",
                        "text": segment.strip(),
                        "start_time_ms": start_ms,
                        "end_time_ms": end_ms,
                        "start_time_srt": start_time_str,
                        "end_time_srt": end_time_str
                    }
                    video_segments.append(segment_info)
                    
                except Exception as e:
                    print(f"  ❌ 处理片段 {j+1} 时出错: {str(e)}")
                    continue

        video_result = {
            "video_id": video_id,
            "original_text": asr_text,
            "total_segments": len(video_segments),
            "segments": video_segments
        }

        safe_video_id = "".join(c for c in video_id if c.isalnum() or c in (' ', '-', '_')).rstrip()
        output_file_path = os.path.join(output_dir, f"{safe_video_id}_segments.json")

        print(f"  正在保存结果到: {output_file_path}")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(video_result, f, ensure_ascii=False, indent=2)
        
        print(f"  ✅ 视频 {video_id} 处理完成，生成 {len(video_segments)} 个有效片段，已保存到 {output_file_path}")
        processed_count += 1
    
    print(f"\n处理完成！共处理 {processed_count} 个视频，结果已分别保存到 {output_dir} 目录中")

if __name__ == "__main__":
    input_json_path = "./annotations/3_all_video_dataset_asr_infer_data.json"
    output_dir = "./annotations/4_video_segments"

    process_video_segments(input_json_path, output_dir)
