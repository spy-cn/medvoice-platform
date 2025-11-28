import logging
import time

import librosa
import numpy as np
import static_ffmpeg
from scipy import signal

static_ffmpeg.add_paths()
from pydub import AudioSegment
from pydub.effects import compress_dynamic_range, normalize
from pydub.scipy_effects import high_pass_filter, low_pass_filter

from src.medvoice.utils.logger_utils import setup_logger

logger = setup_logger(name='AudioUtils', level=logging.DEBUG)

# 定义短音频阈值（单位：毫秒），默认2秒以下为短音频
SHORT_AUDIO_THRESHOLD = 2000

ENHANCEMENT_PARAMS = {
    'compression_threshold': -18.0,  # 稍高阈值，保留更多动态
    'compression_ratio': 3.0,  # 温和压缩
    'high_pass_cutoff': 100,  # 去除更多低频噪声
    'low_pass_cutoff': 4000,  # 语音主要频率范围
    'gain_dB': 6,  # 适当增益
    'noise_reduction_level': 0.03  # 较强降噪
}


def is_short_audio(audio_path, threshold_ms=None):
    """
    判断是否为短音频
    :param audio_path: 音频路径
    :param threshold_ms: 短音频阈值（毫秒）
    :return:
    """
    if threshold_ms is None:
        threshold_ms = SHORT_AUDIO_THRESHOLD

    logger.info(f"开始检测音频是否为短音频: {audio_path}")
    logger.debug(f"使用的阈值: {threshold_ms}ms")
    try:
        start_time = time.time()
        audio = AudioSegment.from_file(audio_path)
        duration_ms = len(audio)
        processing_time = time.time() - start_time

        is_short = duration_ms <= threshold_ms
        logger.info(
            f"音频检测完成 - 路径: {audio_path}, 时长: {duration_ms}ms, 是否为短音频: {is_short}, 处理时间: {processing_time:.3f}s")

        return is_short

    except Exception as e:
        logger.error(f"读取音频文件错误: {audio_path}, 错误信息: {str(e)}", exc_info=True)
        return False


def assess_audio_segment_quality(audio: AudioSegment) -> float:
    """
    综合评估音频质量，返回0-1之间的评分

    评估维度包括：
    - 音量水平
    - 信噪比（简化版）
    - 静音检测
    - 音频失真检测
    - 频率分布

    :param audio: 音频片段
    :return: 质量评分 (0-1，1表示最佳质量)
    """


    # 将音频转换为numpy数组
    samples = np.array(audio.get_array_of_samples())

    # 如果是立体声，取平均值转换为单声道
    if audio.channels == 2:
        samples = samples.reshape(-1, 2)
        samples = np.mean(samples, axis=1)

    # 归一化到[-1, 1]
    if audio.sample_width == 2:
        samples = samples.astype(np.float32) / 32768.0
    elif audio.sample_width == 4:
        samples = samples.astype(np.float32) / 2147483648.0
    else:
        samples = samples.astype(np.float32) / 128.0

    # 1. 音量水平评估
    rms_energy = np.sqrt(np.mean(samples ** 2))
    volume_score = min(rms_energy * 10, 1.0)  # 适当缩放

    # 2. 信噪比评估（简化版）
    # 使用高通滤波器分离信号和噪声成分
    nyquist = audio.frame_rate / 2
    b, a = signal.butter(4, 100 / nyquist, 'high')  # 100Hz高通
    filtered = signal.filtfilt(b, a, samples)

    # 计算信号和噪声的能量
    signal_energy = np.mean(samples ** 2)
    noise_energy = np.mean(filtered ** 2)

    if noise_energy > 0:
        snr = 10 * np.log10(signal_energy / noise_energy)
        snr_score = max(0, min(1, (snr + 10) / 50))  # 将SNR映射到0-1
    else:
        snr_score = 1.0

    # 3. 静音检测
    # 计算短时能量
    frame_length = int(0.02 * audio.frame_rate)  # 20ms帧
    hop_length = frame_length // 2

    energy_frames = []
    for i in range(0, len(samples) - frame_length, hop_length):
        frame = samples[i:i + frame_length]
        energy_frames.append(np.mean(frame ** 2))

    if energy_frames:
        energy_threshold = np.max(energy_frames) * 0.01  # 最大能量的1%作为阈值
        silent_frames = sum(1 for energy in energy_frames if energy < energy_threshold)
        silence_ratio = silent_frames / len(energy_frames)
        silence_score = max(0, 1 - silence_ratio * 2)  # 静音比例越高，分数越低
    else:
        silence_score = 0

    # 4. 削波检测
    clipping_threshold = 0.99
    clipping_samples = np.sum(np.abs(samples) > clipping_threshold)
    clipping_ratio = clipping_samples / len(samples)
    clipping_score = max(0, 1 - clipping_ratio * 10)  # 削波比例越高，分数越低

    # 5. 频率分布评估
    f, Pxx = signal.welch(samples, audio.frame_rate, nperseg=1024)

    # 计算主要频率成分的能量分布
    low_freq_mask = (f >= 80) & (f <= 300)  # 低频
    mid_freq_mask = (f > 300) & (f <= 3000)  # 中频
    high_freq_mask = (f > 3000) & (f <= 8000)  # 高频

    low_energy = np.sum(Pxx[low_freq_mask])
    mid_energy = np.sum(Pxx[mid_freq_mask])
    high_energy = np.sum(Pxx[high_freq_mask])
    total_energy = low_energy + mid_energy + high_energy

    if total_energy > 0:
        # 理想情况下，中频应该占主导
        mid_ratio = mid_energy / total_energy
        frequency_score = min(mid_ratio * 1.5, 1.0)
    else:
        frequency_score = 0

    # 6. 动态范围评估
    dynamic_range = np.max(samples) - np.min(samples)
    dynamic_score = min(dynamic_range * 2, 1.0)

    # 综合评分
    weights = {
        'volume': 0.2,
        'snr': 0.25,
        'silence': 0.2,
        'clipping': 0.15,
        'frequency': 0.1,
        'dynamic': 0.1
    }

    total_score = (
            volume_score * weights['volume'] +
            snr_score * weights['snr'] +
            silence_score * weights['silence'] +
            clipping_score * weights['clipping'] +
            frequency_score * weights['frequency'] +
            dynamic_score * weights['dynamic']
    )

    # 确保分数在0-1范围内
    final_score = max(0, min(1, total_score))

    logger.debug(f"音量评分: {volume_score:.3f}")
    logger.debug(f"信噪比评分: {snr_score:.3f}")
    logger.debug(f"静音评分: {silence_score:.3f}")
    logger.debug(f"削波评分: {clipping_score:.3f}")
    logger.debug(f"频率评分: {frequency_score:.3f}")
    logger.debug(f"动态范围评分: {dynamic_score:.3f}")
    logger.debug(f"最终质量评分: {final_score:.3f}")

    return final_score

def assess_speech_quality(audio: AudioSegment) -> float:
        """
        综合评估音频质量，返回0-1之间的评分
        优化为人声评估，重点关注人声特征

        评估维度包括：
        - 音量水平（针对人声优化）
        - 信噪比（人声频段优化）
        - 静音检测
        - 音频失真检测
        - 人声频率分布
        - 语音清晰度

        :param audio: 音频片段
        :return: 质量评分 (0-1，1表示最佳质量)
        """

        # 将音频转换为numpy数组
        samples = np.array(audio.get_array_of_samples())

        # 如果是立体声，取平均值转换为单声道
        if audio.channels == 2:
            samples = samples.reshape(-1, 2)
            samples = np.mean(samples, axis=1)

        # 归一化到[-1, 1]
        if audio.sample_width == 2:
            samples = samples.astype(np.float32) / 32768.0
        elif audio.sample_width == 4:
            samples = samples.astype(np.float32) / 2147483648.0
        else:
            samples = samples.astype(np.float32) / 128.0

        # 1. 音量水平评估（针对人声优化）
        rms_energy = np.sqrt(np.mean(samples ** 2))
        # 人声理想RMS范围：0.03-0.3，映射到高分区间
        if rms_energy < 0.01:
            volume_score = 0  # 太轻
        elif rms_energy < 0.03:
            volume_score = (rms_energy - 0.01) / 0.02  # 0-0.5
        elif rms_energy <= 0.3:
            volume_score = 0.5 + (rms_energy - 0.03) / 0.54 * 0.4  # 0.5-0.9
        elif rms_energy <= 0.6:
            volume_score = 0.9 - (rms_energy - 0.3) / 0.3 * 0.2  # 0.9-0.7
        else:
            volume_score = 0.7 - min((rms_energy - 0.6) / 0.4, 0.7)  # 0.7-0

        volume_score = max(0, min(1, volume_score))

        # 2. 信噪比评估（人声频段优化）
        nyquist = audio.frame_rate / 2

        # 人声主要频段：85Hz-255Hz（基频）和 300Hz-3400Hz（共振峰）
        vocal_low = [85, 255]  # 基频范围
        vocal_mid = [300, 3400]  # 主要共振峰范围

        # 计算人声频段能量
        b_signal, a_signal = signal.butter(4, [vocal_low[0] / nyquist, vocal_mid[1] / nyquist], 'bandpass')
        vocal_signal = signal.filtfilt(b_signal, a_signal, samples)
        vocal_energy = np.mean(vocal_signal ** 2)

        # 计算噪声频段能量（排除人声主要频段）
        b_noise1, a_noise1 = signal.butter(4, vocal_low[0] / nyquist, 'high')  # 高频噪声
        b_noise2, a_noise2 = signal.butter(4, vocal_mid[1] / nyquist, 'low')  # 低频噪声
        noise_high = signal.filtfilt(b_noise1, a_noise1, samples)
        noise_low = signal.filtfilt(b_noise2, a_noise2, samples)
        noise_energy = (np.mean(noise_high ** 2) + np.mean(noise_low ** 2)) / 2

        if noise_energy > 0 and vocal_energy > 0:
            snr = 10 * np.log10(vocal_energy / noise_energy)
            # 人声SNR理想范围：15-40dB
            if snr < 10:
                snr_score = snr / 10 * 0.6  # 0-0.6
            elif snr <= 40:
                snr_score = 0.6 + (snr - 10) / 30 * 0.4  # 0.6-1.0
            else:
                snr_score = 1.0
        else:
            snr_score = 0.5  # 中性分数

        snr_score = max(0, min(1, snr_score))

        # 3. 静音检测（针对语音优化）
        frame_length = int(0.025 * audio.frame_rate)  # 25ms帧（语音分析常用）
        hop_length = frame_length // 2

        energy_frames = []
        for i in range(0, len(samples) - frame_length, hop_length):
            frame = samples[i:i + frame_length]
            energy_frames.append(np.mean(frame ** 2))

        if energy_frames:
            # 语音静音检测使用自适应阈值
            sorted_energy = np.sort(energy_frames)
            # 使用中位数而不是最大值，避免突发噪声影响
            reference_energy = sorted_energy[len(sorted_energy) // 2]
            energy_threshold = reference_energy * 0.1  # 中位能量的10%作为静音阈值

            silent_frames = sum(1 for energy in energy_frames if energy < energy_threshold)
            silence_ratio = silent_frames / len(energy_frames)

            # 语音中允许一定比例的静音（呼吸、停顿）
            if silence_ratio < 0.1:
                silence_score = 1.0
            elif silence_ratio < 0.3:
                silence_score = 1.0 - (silence_ratio - 0.1) * 2.5  # 1.0-0.5
            elif silence_ratio < 0.7:
                silence_score = 0.5 - (silence_ratio - 0.3) * 1.25  # 0.5-0
            else:
                silence_score = 0
        else:
            silence_score = 0

        # 4. 削波检测
        clipping_threshold = 0.95  # 稍微降低阈值，人声更容易削波
        clipping_samples = np.sum(np.abs(samples) > clipping_threshold)
        clipping_ratio = clipping_samples / len(samples)

        if clipping_ratio == 0:
            clipping_score = 1.0
        elif clipping_ratio < 0.001:  # 0.1%
            clipping_score = 0.8
        elif clipping_ratio < 0.01:  # 1%
            clipping_score = 0.5
        elif clipping_ratio < 0.05:  # 5%
            clipping_score = 0.2
        else:
            clipping_score = 0

        # 5. 人声频率分布评估
        f, Pxx = signal.welch(samples, audio.frame_rate, nperseg=1024)

        # 人声关键频段
        fundamental_mask = (f >= 80) & (f <= 300)  # 基频（男女声基频范围）
        formant_mask = (f > 300) & (f <= 3500)  # 共振峰（语音清晰度关键）
        presence_mask = (f > 3500) & (f <= 6000)  # 临场感
        noise_mask = (f > 6000)  # 高频噪声

        fundamental_energy = np.sum(Pxx[fundamental_mask])
        formant_energy = np.sum(Pxx[formant_mask])
        presence_energy = np.sum(Pxx[presence_mask])
        noise_energy_high = np.sum(Pxx[noise_mask])
        total_energy = fundamental_energy + formant_energy + presence_energy + noise_energy_high

        if total_energy > 0:
            # 理想人声：共振峰能量占比最高，基频适中，高频噪声低
            formant_ratio = formant_energy / total_energy
            fundamental_ratio = fundamental_energy / total_energy
            noise_ratio = noise_energy_high / total_energy

            # 共振峰占比高且高频噪声低时得分高
            frequency_score = min(formant_ratio * 1.8, 1.0)
            # 惩罚过多的高频噪声
            frequency_score *= max(0, 1 - noise_ratio * 3)
            # 基频适中奖励
            if 0.1 <= fundamental_ratio <= 0.4:
                frequency_score = min(frequency_score * 1.1, 1.0)
        else:
            frequency_score = 0

        frequency_score = max(0, min(1, frequency_score))

        # 6. 语音动态范围评估
        # 使用分位数避免极端值影响
        q95 = np.quantile(np.abs(samples), 0.95)
        q5 = np.quantile(np.abs(samples), 0.05)

        if q5 > 0:
            dynamic_range_db = 20 * np.log10(q95 / q5)
            # 语音理想动态范围：30-50dB
            if dynamic_range_db < 20:
                dynamic_score = dynamic_range_db / 20 * 0.7  # 0-0.7
            elif dynamic_range_db <= 50:
                dynamic_score = 0.7 + (dynamic_range_db - 20) / 30 * 0.3  # 0.7-1.0
            else:
                dynamic_score = 1.0
        else:
            dynamic_score = 0

        # 综合评分（针对人声优化权重）
        weights = {
            'volume': 0.25,  # 音量很重要
            'snr': 0.25,  # 信噪比很重要
            'silence': 0.15,  # 适当静音可以接受
            'clipping': 0.15,  # 削波严重影响质量
            'frequency': 0.15,  # 频率分布
            'dynamic': 0.05  # 动态范围次要
        }

        total_score = (
                volume_score * weights['volume'] +
                snr_score * weights['snr'] +
                silence_score * weights['silence'] +
                clipping_score * weights['clipping'] +
                frequency_score * weights['frequency'] +
                dynamic_score * weights['dynamic']
        )

        # 确保分数在0-1范围内
        final_score = max(0, min(1, total_score))

        logger.debug(f"音量评分: {volume_score:.3f}")
        logger.debug(f"信噪比评分: {snr_score:.3f}")
        logger.debug(f"静音评分: {silence_score:.3f}")
        logger.debug(f"削波评分: {clipping_score:.3f}")
        logger.debug(f"频率评分: {frequency_score:.3f}")
        logger.debug(f"动态范围评分: {dynamic_score:.3f}")
        logger.debug(f"最终质量评分: {final_score:.3f}")

        return final_score

def assess_audio_quality(audio_path: str) -> float:
    """
    评估音频质量
    :param audio_path: 需要评估音频的路径
    :return: 音频质量评分
    """
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        quality_metrics = []
        # 1. 音频长度评分
        duration = len(y) / sr
        if duration < 1.0:
            return 0.1
        duration_score = min(duration / 3.0, 1.0)
        quality_metrics.append(duration_score)

        # 2. 信噪比评估
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        if rms_mean < 0.01:
            return 0.2
        snr_score = min(rms_mean * 50, 1.0)
        quality_metrics.append(snr_score)

        # 3. 静音比例
        silence_threshold = 0.01
        silence_ratio = np.mean(rms < silence_threshold)
        speech_ratio = 1.0 - silence_ratio
        quality_metrics.append(speech_ratio)

        # 4. 频谱质心（语音清晰度）
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroids)
        centroid_score = min(spectral_centroid_mean / 4000, 1.0)  # 经验值
        quality_metrics.append(centroid_score)

        # 综合质量评分（加权平均）
        weights = [0.3, 0.3, 0.2, 0.2]  # 时长和信噪比权重更高
        overall_quality = np.average(quality_metrics, weights=weights)

        return float(overall_quality)
    except Exception as e:
        logger.error(f"评估音频质量失败: {e}")
        return 0.3

def save_audio(save_audio_segment: AudioSegment, output_path: str):
    logger.info("开始保存音频")
    try:
        start_time = time.time()
        audio_format = output_path.split(".")[-1]
        save_audio_segment.export(output_path, format=audio_format)
        save_time = time.time() - start_time
        logger.info(f"音频文件保存成功 - 路径: {output_path}, 保存时间: {save_time:.3f}s")
    except Exception as e:
        logger.error(f"保存音频文件失败: {output_path}, 错误信息: {str(e)}", exc_info=True)
        raise Exception(f"保存音频文件失败: {e}")


def repeat_audio_pydub_exact(audio: AudioSegment, target_duration_ms=1000) -> AudioSegment:
    """
    重复音频整数次，使总时长最接近但不小于目标时长
    :param audio: 输入的音频片段对象
    :param target_duration_ms: 目标时长（毫秒），默认为1000ms（1秒）
    :return: 重复后的音频片段
    """
    original_duration = len(audio)

    # 如果原始音频已经达到或超过目标时长，直接返回
    if original_duration >= target_duration_ms:
        return audio

    # 计算需要重复的最小次数（使总时长 >= 目标时长）
    repeat_times = (target_duration_ms + original_duration - 1) // original_duration

    return audio * repeat_times


def basic_enhancement(audio: AudioSegment = None):
    """
    对基础音频进行增强处理
    :param audio:  AudioSegment对象
    :return:
    """
    # 验证输入音频
    if audio is None or len(audio) == 0:
        logger.warning("输入音频为空，跳过增强处理")
        return audio
    start_time = time.time()
    logger.debug("开始基础音频增强处理")
    original_duration = len(audio)
    enhanced_audio = audio
    try:
        logger.debug(
            f"原始音频信息 - 时长: {original_duration}ms, 帧率: {audio.frame_rate}Hz, 声道数: {audio.channels}")
        # 1. 动态范围压缩（先压缩动态范围，避免后续处理引入失真）
        logger.debug(
            f"应用动态范围压缩, 阈值: {ENHANCEMENT_PARAMS['compression_threshold']}dB, "
            f"比率: {ENHANCEMENT_PARAMS['compression_ratio']}:1"
        )
        enhanced_audio = compress_dynamic_range(
            enhanced_audio,
            threshold=ENHANCEMENT_PARAMS['compression_threshold'],
            ratio=ENHANCEMENT_PARAMS['compression_ratio']
        )
        # 2. 增益调整（在滤波前适当提升音量）
        if ENHANCEMENT_PARAMS['gain_dB'] != 0:
            logger.debug(f"应用增益调整: {ENHANCEMENT_PARAMS['gain_dB']:+}dB")
            enhanced_audio = enhanced_audio + ENHANCEMENT_PARAMS['gain_dB']

        # 3. 高通滤波去除低频噪声（如嗡嗡声、风声）
        if ENHANCEMENT_PARAMS['high_pass_cutoff'] > 0:
            logger.debug(f"应用高通滤波, 截止频率: {ENHANCEMENT_PARAMS['high_pass_cutoff']}Hz")
            enhanced_audio = high_pass_filter(
                enhanced_audio,
                ENHANCEMENT_PARAMS['high_pass_cutoff']
            )

        # 4. 低通滤波去除高频噪声（如嘶嘶声、电流声）
        if ENHANCEMENT_PARAMS['low_pass_cutoff'] > 0:
            logger.debug(f"应用低通滤波, 截止频率: {ENHANCEMENT_PARAMS['low_pass_cutoff']}Hz")
            enhanced_audio = low_pass_filter(
                enhanced_audio,
                ENHANCEMENT_PARAMS['low_pass_cutoff']
            )

        # 5. 音量标准化（最后统一音量水平）
        logger.debug("应用音量标准化")
        enhanced_audio = normalize(enhanced_audio)

        processing_time = time.time() - start_time
        final_duration = len(enhanced_audio)

        logger.info(
            f"基础音频增强处理完成 - "
            f"原始时长: {original_duration}ms, 最终时长: {final_duration}ms, "
            f"处理时间: {processing_time:.3f}s"
        )

        # 验证输出音频
        if final_duration == 0:
            logger.error("增强后音频时长为0，返回原始音频")
            return audio

        return enhanced_audio
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(
            f"基础增强处理过程中出现错误: {str(e)}",
            exc_info=True
        )
        logger.warning(f"基础增强处理失败，返回原始音频，处理时间: {processing_time:.3f}s")
        return audio


if __name__ == '__main__':
    # audio_path = r"E:\code_project\medvoice-recognition-platform\data\audio\voiceprint_lib\叶问老婆1.wav"
    # audio = AudioSegment.from_file(audio_path)
    # logger.info("开始执行")
    # save_audio_segment = basic_enhancement(audio)
    # save_audio_path = r"E:\code_project\medvoice-recognition-platform\data\audio\enhance_audio\叶问老婆1.wav"
    # save_audio(save_audio_segment, save_audio_path)
    # quality = assess_audio_quality(
    #     r"E:\code_project\medvoice-recognition-platform\data\audio\enhance_audio\叶问老婆1.wav")
    # print(quality)
    # quality_enhance_audio = assess_audio_quality(
    #     r"E:\code_project\medvoice-recognition-platform\data\audio\voiceprint_lib\叶问老婆2.wav")
    # print(quality_enhance_audio)
    #重复音频
    need_repeat_audio = r"E:\code_project\medvoice-recognition-platform\data\audio\segments\segment_0_0_510_750.wav"
    audio = AudioSegment.from_file(need_repeat_audio)
    repeat_audio = repeat_audio_pydub_exact(audio,2000)
    save_audio(repeat_audio, r"E:\code_project\medvoice-recognition-platform\data\audio\enhance_audio\repeat_segment_0_0_510_750.wav")
    # 检查语音质量
    audio_path=r"E:\code_project\medvoice-recognition-platform\data\audio\enhance_audio\repeat_segment_0_0_510_750.wav"
    audio_segment = AudioSegment.from_file(audio_path)
    quality = assess_speech_quality(audio_segment)
    logger.debug(f"短语音的质量：{quality}")

