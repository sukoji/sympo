import torch
import whisperx
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import gc

def process_meeting_audio(audio_file, hf_token):
    # GPU 사용 가능 여부 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4 
    compute_type = "int8" # VRAM 절약을 위해 int8 사용 
    
    print(f"[{device}] 환경에서 실행 중입니다...")

    # 1. 오디오 음성 인식 (STT - Whisper)
    print("\n[1/4] Whisper 모델로 음성을 텍스트로 변환하는 중...")
    model = whisperx.load_model("large-v3", device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()
    
    # 2. 단어/문장 타이밍 정렬 (Alignment)
    print("\n[2/4] 화자 구분을 위해 오디오 타이밍을 정밀하게 맞추는 중...")
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    del model_a
    torch.cuda.empty_cache()
    gc.collect()
    
    # 3. 화자 분리 (Speaker Diarization - Pyannote)
    print("\n[3/4] 화자(Speaker)를 구분하는 중 (Diarization)...")
    from whisperx.diarize import DiarizationPipeline
    diarize_model = DiarizationPipeline(token=hf_token, device=device)
    diarize_segments = diarize_model(audio_file)
    del diarize_model
    torch.cuda.empty_cache()
    gc.collect()
    
    # 음성 인식 결과에 화자 매핑
    result = whisperx.assign_word_speakers(diarize_segments, result)
    
    # 텍스트화된 녹취록 생성
    transcript = ""
    for segment in result["segments"]:
        # 화자가 인식되지 않은 경우 Unknown으로 표시
        speaker = segment.get("speaker", "Unknown")
        text = segment.get("text", "").strip()
        transcript += f"[{speaker}] : {text}\n"
        
    print("\n--- 화자 분리 완료된 녹취록 ---\n")
    print(transcript)
    print("\n------------------------------\n")
    
    return transcript
    
#     # 4. Gemma 모델을 사용해 회의 내용 분석 및 요약
#     print("\n[4/4] Gemma 모델로 회의 내용을 분석하는 중...")
    
#     # 반가운 소식이네요! 2026년 4월에 출시된 최신 Gemma 4 모델을 사용합니다.
#     # Gemma 4는 E2B, E4B, 26B(MoE), 31B 모델 크기로 제공됩니다.
#     # 여기서는 적절한 크기의 e4b 모델을 사용하도록 지정했습니다.
#     model_id = "google/gemma-4-e4b-it" 
    
#     tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, extra_special_tokens={})
#     gemma_model = AutoModelForCausalLM.from_pretrained(
#         model_id,
#         device_map="auto",
#         torch_dtype=torch.float16 if device == "cuda" else torch.float32,
#         token=hf_token
#     )
    
#     prompt = f"""다음은 여러 명이 참석한 회의의 녹취록입니다.
# 아래의 지시사항에 따라 회의 내용을 정리해주세요:
# 1. 전체 회의의 핵심 주제 요약
# 2. 각 화자(Speaker)별 주요 발언 내용 요약
# 3. 회의 이후 진행해야 할 액션 아이템(Action Items) 도출

# 회의 녹취록:
# {transcript}

# 정리 내용:"""
    
#     inputs = tokenizer(prompt, return_tensors="pt").to(gemma_model.device)
    
#     # 생성 속도 및 토큰 길이 설정
#     outputs = gemma_model.generate(
#         **inputs,
#         max_new_tokens=1500,
#         temperature=0.3,
#         do_sample=True
#     )
    
#     response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    
#     print("\n--- Gemma 분석 결과 ---\n")
#     print(response)

# if __name__ == "__main__":
#     AUDIO_FILE_PATH = "/home/piai/TEST/AI/temp_1775718059373.242133713.m4a" 
#     process_meeting_audio(AUDIO_FILE_PATH, os.environ.get("HF_TOKEN"))

