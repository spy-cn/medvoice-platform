import logging
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import static_ffmpeg
from funasr import AutoModel
from pydub import AudioSegment
from sklearn.metrics.pairwise import cosine_similarity

from src.medvoice.utils import audio_utils
from src.medvoice.utils.logger_utils import setup_logger

static_ffmpeg.add_paths()

warnings.filterwarnings('ignore')

logger = setup_logger('SpeakerIdentification', level=logging.DEBUG)

project_root = Path(__file__).resolve().parents[3]  # 调整层级
sys.path.append(str(project_root))


@dataclass
class SpeakerSegment:
    """说话人音频片段数据类"""
    start_time: float  # 开始时间（毫秒）
    end_time: float  # 结束时间（毫秒）
    text: str  # 识别文本
    spk_code: str  # 说话人编码
    spk_name: str  # 说话人姓名
    speaker_id: str  # 说话人ID
    audio_path: str  # 音频文件路径
    similarity: float  # 相似度得分


class SpeakerIdentification:
    def __init__(self):
        self.asr_model = None
        self.sv_model = None
        self.speaker_profiles = {}  # 存储已知说话人的声纹特征
        self.speaker_names = {}  # 说话人ID到姓名的映射
        self.speaker_counter = 1  # 说话人计数器
        self.similarity_threshold = 0.7  # 相似度阈值
        self.temp_dir = r"E:\code_project\medvoice-recognition-platform\data\audio"

    def init_models(self):
        """初始化所有模型"""
        try:
            # 初始化ASR模型（包含说话人分离）
            self.asr_model = AutoModel(
                model="paraformer-zh",
                vad_model="fsmn-vad",
                punc_model="ct-punc",
                spk_model="cam++",
                disable_update=True
            )
            logger.info("✅ ASR模型加载成功")

            # 初始化声纹识别模型
            self.sv_model = AutoModel(
                model="cam++",
                disable_update=True
            )
            logger.info("✅ 声纹识别模型加载成功")
            return True

        except Exception as e:
            logger.error(f"❌ 模型初始化失败: {e}")
            return False

    def _extract_voiceprint_embedding(self, audio_path: str) -> Optional[np.ndarray]:
        """提取声纹嵌入向量"""
        try:
            if not os.path.exists(audio_path):
                logger.error(f"音频文件不存在: {audio_path}")
                return None

            # 音频质量检测和增强
            audio_segment = AudioSegment.from_file(audio_path)
            audio_quality = audio_utils.assess_speech_quality(audio_segment)

            if audio_quality < 0.5:
                logger.debug(f"音频质量较低({audio_quality:.3f})，进行增强: {audio_path}")
                enhanced_audio = self._enhance_audio_quality(audio_segment, target_duration=2000)
                enhanced_path = audio_path.replace('.wav', '_enhanced.wav')
                enhanced_audio.export(enhanced_path, format="wav")
                audio_path = enhanced_path  # 使用增强后的音频

            # 提取声纹特征
            result = self.sv_model.generate(input=audio_path)

            # 清理临时增强文件
            if '_enhanced' in audio_path and os.path.exists(audio_path):
                os.remove(audio_path)

            return self._process_embedding_result(result)

        except Exception as e:
            logger.error(f"提取声纹特征失败 {audio_path}: {e}")
            return None

    def _enhance_audio_quality(self, audio: AudioSegment, target_duration: int, max_attempts: int = 3) -> AudioSegment:
        """音频质量增强"""
        enhanced_audio = audio_utils.repeat_audio_pydub_exact(audio, target_duration)
        quality_score = 0.0
        for attempt in range(1, max_attempts + 1):
            quality_score = audio_utils.assess_speech_quality(enhanced_audio)
            if quality_score >= 0.5:
                logger.debug(f"✅ 音频增强成功 (第{attempt}次尝试, 质量: {quality_score:.3f})")
                return enhanced_audio
            elif attempt < max_attempts:
                logger.debug(f"🔄 继续音频增强 (第{attempt}次尝试, 质量: {quality_score:.7f})")
                enhanced_audio = audio_utils.repeat_audio_pydub_exact(enhanced_audio, target_duration * attempt)

        logger.warning(f"⚠️ 音频增强未达理想质量 (最终质量: {quality_score:.7f})")
        return enhanced_audio

    def _process_embedding_result(self, result):

        if result and isinstance(result, list) and len(result) > 0:
            embedding = None

            if 'spk_embedding' in result[0]:
                embedding_tensor = result[0]['spk_embedding']
                embedding = embedding_tensor.cpu().numpy() if hasattr(embedding_tensor, 'cpu') else np.array(
                    embedding_tensor)

            if embedding is not None:
                # 标准化维度
                if len(embedding.shape) == 1:
                    embedding = embedding.reshape(1, -1)
                elif len(embedding.shape) == 2 and embedding.shape[0] > 1:
                    embedding = embedding[0].reshape(1, -1)

                # L2归一化
                embedding = embedding / np.linalg.norm(embedding)
            return embedding
        return None

    def collect_speaker_voiceprints(self,
                                    speaker_name: str, audio_paths: List[str], min_audio_count: int = 3,
                                    quality_threshold: float = 0.4) -> Optional[str]:
        """
        收集用户声纹信息
        :param speaker_name: 说话人姓名
        :param audio_paths: 音频文件路径列表
        :param min_audio_count: 最少需要的合格音频数量
        :param quality_threshold: 音频质量阈值
        :return: 说话人ID 或Nonde
        """
        logger.info(f"开始为{speaker_name}收集声纹信息...")
        logger.info(f"待处理音频数量：{len(audio_paths)}")
        collect_embeddings = []
        quality_scores = []

        for i, audio_path in enumerate(audio_paths):
            if not os.path.exists(audio_path):
                logger.error(f"音频文件不存在：{audio_path}")
                continue
            try:
                # 1、先检查音频质量
                need_audio_segment = AudioSegment.from_file(audio_path)
                audio_quality_score = audio_utils.assess_speech_quality(need_audio_segment)
                logger.debug(f"音频质量频分为：{audio_quality_score}")
                # 如果音频质量评分不达标 进行增强
                if audio_quality_score < quality_threshold:
                    audio = AudioSegment.from_file(audio_path)
                    repeat_audio = audio_utils.repeat_audio_pydub_exact(audio, 2000)
                    enhancement_attempt = 0
                    max_enhancement_attempts = 2
                    while enhancement_attempt <= max_enhancement_attempts:
                        quality_score = audio_utils.assess_speech_quality(repeat_audio)
                        if quality_score > quality_threshold:
                            break
                        else:
                            if enhancement_attempt < max_enhancement_attempts:
                                # 再次增强
                                enhancement_attempt += 1
                                logger.debug(
                                    f"🔄 尝试第 {enhancement_attempt} 次重新增强低质量音频: {os.path.basename(audio_path)}")
                            else:
                                logger.debug(
                                    f"⚠️ 低质量音频(已达最大增强次数): {max_enhancement_attempts} (质量: {quality_score})")
                                break
                else:
                    quality_scores.append(audio_quality_score)
                # 2、提取声纹信息
                embedding = self._extract_voiceprint_embedding(audio_path)
                collect_embeddings.append(embedding)
            except Exception as e:
                logger.error(e)

        if len(collect_embeddings) < min_audio_count:
            logger.error(f"有效声纹数量不足，需要至少 {min_audio_count} 个，当前 {len(collect_embeddings)} 个")
            return None
        # 3、质量加权平均融合
        logger.debug("正在进行声纹特征融合...")
        combined_embedding = self._fuse_voiceprints(collect_embeddings, quality_scores)
        # 4、注册说话人
        spk_code = self._register_or_update_speaker(speaker_name, combined_embedding)
        logger.info(f"成功为 '{speaker_name}' 注册声纹，CODE: {spk_code}")
        return spk_code

    def process_audio_with_spk_diarization(self, audio_path: str, hotword: str = None) -> List[SpeakerSegment]:
        """
        处理音频并进行说话人分离和识别
        :param audio_path: 要处理的音频路径
        :param hotword: 热词
        :return:
        """
        if not self.asr_model:
            logger.error("ASR模型未初始化")
            return []

        if not os.path.exists(audio_path):
            logger.error(f"音频文件不存在: {audio_path}")
            return []

        logger.info(f"开始处理音频: {audio_path}")
        try:
            # 说话人分离
            res = self.asr_model.generate(
                input=audio_path,
                batch_size_s=300,
                hotword=hotword
            )
            if not res:
                logger.error("没有识别到有效语音")
                return []
            # 处理每一个语音片段
            speaker_segments = []
            audio = AudioSegment.from_file(audio_path)
            for i, segment in enumerate(res):
                sentence_info_list = segment.get('sentence_info', [])
                for j, sentence_info in enumerate(sentence_info_list):
                    segment_result = self._process_single_segment(
                        sentence_info, audio, i, j
                    )
                    if segment_result:
                        speaker_segments.append(segment_result)
            # 识别说话人
            identified_segments = self._identify_speakers_in_segments(speaker_segments)

            self._print_recognition_results(identified_segments)
            return identified_segments
        except Exception as e:
            logger.error(f"处理音频失败：{e}")
            return []

        pass

    def _fuse_voiceprints(self, embeddings: List[np.ndarray], quality_scores: List[float]) -> np.ndarray:
        """
        融合多个声纹特征
        :param embeddings:
        :param quality_scores:
        :return:
        """
        # 归一化质量权重
        weights = np.array(quality_scores) / sum(quality_scores)

        # 计算加权平均
        combined_embedding = np.average(embeddings, axis=0, weights=weights)

        # L2归一化
        combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)

        return combined_embedding

    def _register_or_update_speaker(self, speaker_name: str, embedding: np.ndarray) -> str:
        """
        注册或者更新说话人
        :param speaker_name:  说话人姓名
        :param embedding:  声纹向量
        :return:
        """
        # 检查是否已经存在此说话人
        existing_code = None
        for spk_code, name in self.speaker_names.items():
            if name == speaker_name:
                existing_code = spk_code
                break
        if existing_code:
            # 更新现有说话人（指数平滑更新）
            old_embedding = self.speaker_profiles[existing_code]
            updated_embedding = 0.7 * embedding + 0.3 * old_embedding
            updated_embedding = updated_embedding / np.linalg.norm(updated_embedding)

            self.speaker_profiles[existing_code] = updated_embedding
            logger.info(f"已更新说话人: {speaker_name} (CODE: {existing_code})")
            return existing_code
        else:
            # 新注册
            spk_code = f"spk_{self.speaker_counter:03d}"
            self.speaker_counter += 1
            self.speaker_profiles[spk_code] = embedding
            self.speaker_names[spk_code] = speaker_name
            logger.info(f"已注册新说话人: {speaker_name} (CODE: {spk_code})")
            return spk_code

    def _process_single_segment(self, sentence_info: dict, audio: AudioSegment,
                                segment_idx: int, sentence_idx: int) -> Optional[SpeakerSegment]:
        """
        处理单个语音片段
        :param sentence_info:
        :param audio:
        :param segment_idx:
        :param sentence_idx:
        :return:
        """
        try:
            segments_dir = os.path.join(self.temp_dir, "segments")
            os.makedirs(segments_dir, exist_ok=True)

            text = sentence_info.get('text', '')
            start_time = sentence_info.get('start', 0)
            end_time = sentence_info.get('end', 0)
            spk_id = sentence_info.get('spk', '未知')
            logger.debug(f"说话人：{spk_id}")
            # 提取音频片段
            segment_audio = audio[start_time:end_time]
            # 保存片段
            segment_filename = f"segment_{segment_idx}_{sentence_idx}_{start_time}_{end_time}.wav"
            segment_path = os.path.join(self.temp_dir, "segments", segment_filename)
            logger.debug(f"保存的文件路径：{segment_path}")
            segment_audio.export(segment_path, format="wav")
            return SpeakerSegment(
                start_time=start_time,
                end_time=end_time,
                text=text,
                speaker_id=spk_id,
                spk_code="",
                spk_name=spk_id,
                audio_path=segment_path,
                similarity=0.0
            )
        except Exception as e:
            logger.error(f"处理语音片段失败:{e}")
            return None

    def _identify_speakers_in_segments(self, speaker_segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        """
        识别片段中的说话人
        :param speaker_segments:
        :return:
        """
        identified_segments = []
        spk_map = {}
        max_similarity_map = {}
        for segment in speaker_segments:

            query_embedding = self._extract_voiceprint_embedding(segment.audio_path)
            if query_embedding is None:
                logger.warning(f"未识别到说话人ID:{segment.audio_path}")
                segment.speaker_id = "unknown"
                identified_segments.append(segment)
                continue
            duration = segment.end_time - segment.start_time
            dynamic_threshold = self._get_dynamic_threshold(duration)
            best_match_spk_code, best_score = self._match_against_voiceprint_library(query_embedding, dynamic_threshold)
            if best_match_spk_code:
                spk_map[segment.speaker_id] = best_match_spk_code
                segment.spk_code = best_match_spk_code
                segment.similarity = best_score
                # 更新spk_map，只保留相似度最大的映射
                if segment.speaker_id and segment.speaker_id != "unknown":
                    current_max_similarity = max_similarity_map.get(segment.speaker_id, -1)
                    if best_score > current_max_similarity:
                        spk_map[segment.speaker_id] = best_match_spk_code
                        max_similarity_map[segment.speaker_id] = best_score
                        logger.debug(f"更新映射: {segment.speaker_id} -> {best_match_spk_code}, 相似度: {best_score}")
            else:
                segment.spk_code = spk_map.get(segment.speaker_id, "unknown")
                segment.similarity = best_score
            identified_segments.append(segment)
        print(f"======================:{spk_map}")
        return identified_segments

    def _get_dynamic_threshold(self, duration: float) -> float:
        """根据音频时长动态调整相似度阈值"""
        base_threshold = self.similarity_threshold

        if duration < 1000:  # 少于1秒
            return max(0.5, base_threshold - 0.2)
        elif duration < 2000:  # 1-2秒
            return max(0.6, base_threshold - 0.1)
        else:  # 2秒以上
            return base_threshold

    def _match_against_voiceprint_library(self, query_embedding: np.ndarray,
                                          threshold: float) -> Tuple[Optional[str], float]:
        """与声纹库进行匹配"""
        best_match_spk_code = None
        best_score = 0.0

        for spk_code, profile_embedding in self.speaker_profiles.items():
            try:
                # 确保维度一致
                if query_embedding.shape[1] != profile_embedding.shape[1]:
                    continue

                # 计算余弦相似度
                similarity = cosine_similarity(query_embedding, profile_embedding)[0][0]
                logger.debug(f"说话人code：{spk_code},相似度：{similarity}")
                if similarity > best_score and similarity >= threshold:
                    best_score = similarity
                    best_match_spk_code = spk_code

            except Exception as e:
                logger.error(f"与说话人 {spk_code} 比对失败: {e}")
                continue

        return best_match_spk_code, best_score

    def _print_recognition_results(self, segments: List[SpeakerSegment]):
        """打印识别结果"""
        logger.debug("\n" + "=" * 60)
        logger.debug("说话人识别结果")
        logger.debug("=" * 60)

        for segment in segments:
            speaker_name = self.speaker_names.get(segment.spk_code, "未知说话人")

            if segment.spk_code != "unknown":
                logger.debug(f"✅ 识别到: {speaker_name} (相似度: {segment.similarity:.3f})")
            else:
                logger.debug(f"❌ 未识别 (最高相似度: {segment.similarity:.3f})")
            logger.debug(f"说话人ID：: {segment.speaker_id}")
            logger.debug(f"说话人CODE：: {segment.spk_code}")
            logger.debug(f"时间: {segment.start_time / 1000:.2f}s - {segment.end_time / 1000:.2f}s")
            logger.debug(f"内容: {segment.text}")
            logger.debug(f"音频路径: {segment.audio_path}")
            logger.debug("-" * 40)


if __name__ == "__main__":
    spkIdent = SpeakerIdentification()
    spkIdent.init_models()
    speakers_data = {
        "叶问老婆": [
            r'E:\code_project\medvoice-recognition-platform\data\audio\voiceprint_lib\叶问老婆1.wav',
            r'E:\code_project\medvoice-recognition-platform\data\audio\voiceprint_lib\叶问老婆2.wav',
            r'E:\code_project\medvoice-recognition-platform\data\audio\voiceprint_lib\叶问老婆3.wav',
            r'E:\code_project\medvoice-recognition-platform\data\audio\voiceprint_lib\叶问老婆4.wav',
        ],
        # "叶问": [
        #     r'E:\code_project\medvoice-recognition-platform\data\audio\voiceprint_lib\叶问1.wav',
        #     r'E:\code_project\medvoice-recognition-platform\data\audio\voiceprint_lib\叶问2.wav',
        #     r'E:\code_project\medvoice-recognition-platform\data\audio\voiceprint_lib\叶问3.wav',
        # ],
        "叶问讲解者": [
            r'E:\code_project\medvoice-recognition-platform\data\audio\voiceprint_lib\叶问讲解者1.wav',
            r'E:\code_project\medvoice-recognition-platform\data\audio\voiceprint_lib\叶问讲解者2.wav',
            r'E:\code_project\medvoice-recognition-platform\data\audio\voiceprint_lib\叶问讲解者3.wav',
            r'E:\code_project\medvoice-recognition-platform\data\audio\voiceprint_lib\叶问讲解者4.wav',
            r'E:\code_project\medvoice-recognition-platform\data\audio\voiceprint_lib\叶问讲解者5.wav',
            r'E:\code_project\medvoice-recognition-platform\data\audio\voiceprint_lib\叶问讲解者6.wav',
        ],
        "路人": [
            r'E:\code_project\medvoice-recognition-platform\data\audio\voiceprint_lib\路人1.wav',
            r'E:\code_project\medvoice-recognition-platform\data\audio\voiceprint_lib\路人2.wav',
        ],
        "金山爪": [
            r'E:\code_project\medvoice-recognition-platform\data\audio\voiceprint_lib\金山爪0.wav',
            r'E:\code_project\medvoice-recognition-platform\data\audio\voiceprint_lib\金山爪1.wav',
            r'E:\code_project\medvoice-recognition-platform\data\audio\voiceprint_lib\金山爪2.wav',
            r'E:\code_project\medvoice-recognition-platform\data\audio\voiceprint_lib\金山爪3.wav',
            r'E:\code_project\medvoice-recognition-platform\data\audio\voiceprint_lib\金山爪.wav',
        ]
    }
    for speaker_name, audio_paths in speakers_data.items():
        spk_code = spkIdent.collect_speaker_voiceprints(speaker_name, audio_paths, min_audio_count=2,
                                                        quality_threshold=0.4)
        if spk_code:
            logger.info(f"✅ 成功注册: {speaker_name} -> {spk_code}")
        else:
            logger.error(f"❌ 注册失败: {speaker_name}")
    target_audio = r"E:\code_project\medvoice-recognition-platform\data\audio\origin_audio\叶问.wav"
    if os.path.exists(target_audio):
        segments = spkIdent.process_audio_with_spk_diarization(target_audio)
        logger.debug(f"处理完成，共识别 {len(segments)} 个语音片段")
    else:
        logger.debug(f"目标音频文件不存在: {target_audio}")
