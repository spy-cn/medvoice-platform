import logging
import subprocess
from pathlib import Path
from typing import Optional
from logger_utils import setup_logger
import static_ffmpeg

static_ffmpeg.add_paths()


logger = setup_logger(name='VideoToAudio')


class VideoToAudio:
    SUPPORTED_VIDEO_FORMATS = {
        '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv',
        '.webm', '.m4v', '.3gp', '.ts', '.mts', '.m2ts'
    }

    # 扩展音频格式配置，提供更多选项
    SUPPORTED_AUDIO_FORMATS = {
        'mp3': {'codec': 'libmp3lame', 'bitrate': '192k', 'default_ext': 'mp3'},
        'wav': {'codec': 'pcm_s16le', 'bitrate': None, 'default_ext': 'wav'},
        'aac': {'codec': 'aac', 'bitrate': '192k', 'default_ext': 'aac'},
        'flac': {'codec': 'flac', 'bitrate': None, 'default_ext': 'flac'},
        'ogg': {'codec': 'libvorbis', 'bitrate': '192k', 'default_ext': 'ogg'},
        'm4a': {'codec': 'aac', 'bitrate': '192k', 'default_ext': 'm4a'},
        'wma': {'codec': 'wmav2', 'bitrate': '192k', 'default_ext': 'wma'}
    }

    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        self.ffmpeg_path = ffmpeg_path
        self._check_ffmpeg_availability()

    def _check_ffmpeg_availability(self) -> bool:
        """检查FFmpeg是否可用"""
        try:
            result = subprocess.run(
                [self.ffmpeg_path, '-version'],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"FFmpeg版本: {result.stdout.split()[2]}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"FFmpeg不可用: {e}")
            raise RuntimeError(f"FFmpeg不可用: {e}")

    def _validate_input_video(self, video_path: str) -> bool:
        """验证输入视频文件"""
        video_path = Path(video_path)
        if not video_path.exists():
            logger.error(f"视频文件不存在: {video_path}")
            return False
        if not video_path.is_file():
            logger.error(f"路径不是文件: {video_path}")
            return False

        suffix = video_path.suffix.lower()
        if suffix not in self.SUPPORTED_VIDEO_FORMATS:
            logger.warning(f"不常见的视频格式: {suffix}，尝试继续处理")

        return True

    def _get_output_path(self, video_path: str, output_format: str,
                         output_dir: Optional[str] = None) -> str:
        """生成输出文件路径"""
        video_path = Path(video_path)

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = video_path.parent

        # 使用配置中的默认扩展名
        default_ext = self.SUPPORTED_AUDIO_FORMATS[output_format.lower()]['default_ext']
        output_filename = f"{video_path.stem}.{default_ext}"
        output_path = output_dir / output_filename

        logger.info(f"输出文件：{output_path}")
        return str(output_path)

    def extract_audio(self,
                      video_path: str,
                      output_format: str = "wav",
                      output_path: Optional[str] = None,
                      overwrite: bool = True,
                      audio_bitrate: Optional[str] = None,
                      sample_rate: Optional[int] = None,
                      audio_channels: Optional[int] = None) -> Optional[str]:
        """
        从视频中提取音频

        :param video_path: 输入视频路径
        :param output_format: 输出音频格式
        :param output_path: 输出文件路径（可选）
        :param overwrite: 是否覆盖已存在文件
        :param audio_bitrate: 音频比特率（可选）
        :param sample_rate: 采样率（可选）
        :param audio_channels: 声道数（可选）
        :return: 成功返回输出路径，失败返回None
        """
        try:
            # 验证输入文件
            if not self._validate_input_video(video_path):
                return None

            # 验证输出格式
            output_format_lower = output_format.lower()
            if output_format_lower not in self.SUPPORTED_AUDIO_FORMATS:
                logger.error(f"不支持的音频格式: {output_format}")
                logger.info(f"支持的格式: {', '.join(self.SUPPORTED_AUDIO_FORMATS.keys())}")
                return None

            # 生成输出路径
            if output_path is None:
                output_path = self._get_output_path(video_path, output_format_lower)

            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)

            # 检查输出文件是否已存在
            if output_path_obj.exists() and not overwrite:
                logger.error(f"输出文件已存在: {output_path}，使用overwrite=True覆盖")
                return None

            # 构建FFmpeg命令
            audio_config = self.SUPPORTED_AUDIO_FORMATS[output_format_lower]
            cmd = [
                self.ffmpeg_path,
                '-i', str(video_path),
                '-vn',  # 不处理视频流
                '-acodec', audio_config['codec'],
            ]

            # 添加可选参数
            if audio_bitrate:
                cmd.extend(['-ab', audio_bitrate])
            elif audio_config['bitrate']:
                cmd.extend(['-ab', audio_config['bitrate']])

            if sample_rate:
                cmd.extend(['-ar', str(sample_rate)])
            else:
                cmd.extend(['-ar', '44100'])  # 默认采样率

            if audio_channels:
                cmd.extend(['-ac', str(audio_channels)])
            else:
                cmd.extend(['-ac', '2'])  # 默认立体声

            # 覆盖选项
            cmd.extend(['-y' if overwrite else '-n'])
            cmd.append(str(output_path))

            logger.info(f"开始提取音频: {video_path} -> {output_path}")
            logger.debug(f"FFmpeg命令: {' '.join(cmd)}")

            # 执行命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            # 检查输出文件是否生成
            if not output_path_obj.exists():
                logger.error("输出文件未生成，提取可能失败")
                return None

            file_size = output_path_obj.stat().st_size
            logger.info(f"音频提取成功: {output_path} (大小: {file_size} 字节)")
            return str(output_path)

        except subprocess.CalledProcessError as e:
            logger.error(f"音频提取失败: {e}")
            logger.error(f"FFmpeg错误输出: {e.stderr}")
            return None
        except Exception as e:
            logger.error(f"发生未知错误: {e}")
            return None


# 使用示例
if __name__ == "__main__":
    try:
        video_to_audio = VideoToAudio()
        result = video_to_audio.extract_audio(
            video_path=r'/Users/spy/Documents/codes/python_code/medvoice-platform/data/video/叶问.mp4',
            output_path=r'/Users/spy/Documents/codes/python_code/medvoice-platform/data/audio/origin_audio/叶问.wav',
            output_format='wav',
            sample_rate=48000,  # 自定义采样率
            audio_channels=2  # 单声道
        )
        if result:
            print(f"提取成功: {result}")
        else:
            print("提取失败")

    except Exception as e:
        print(f"初始化失败: {e}")