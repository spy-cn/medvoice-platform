import logging
import pymysql
from typing import List, Dict, Any, Optional, Union
import logging
from contextlib import contextmanager
import json

from src.medvoice.utils.logger_utils import setup_logger

logger = setup_logger(name='MySQLConnectionUtil', level=logging.DEBUG)


class MySQLConnectionUtil:
    def __init__(self, host: str = 'localhost', port: int = 3306,
                 user: str = 'root', password: str = 'password',
                 database: str = '语音识别', charset: str = 'utf8mb4'):
        """
        初始化数据库连接参数

        Args:
            host: 数据库主机地址
            port: 数据库端口
            user: 用户名
            password: 密码
            database: 数据库名
            charset: 字符集
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.charset = charset
        self.connection = None

    def connect(self) -> bool:
        """
        连接到数据库

        Returns:
            bool: 连接是否成功
        """
        try:
            self.connection = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                charset=self.charset,
                cursorclass=pymysql.cursors.DictCursor  # 返回字典格式的结果
            )
            logger.info(f"成功连接到数据库: {self.database}")
            return True
        except pymysql.Error as e:
            logger.error(f"数据库连接失败: {e}")
            return False

    def disconnect(self):
        """断开数据库连接"""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("数据库连接已关闭")

    def is_connected(self) -> bool:
        """检查连接状态"""
        if self.connection:
            try:
                self.connection.ping(reconnect=True)
                return True
            except pymysql.Error:
                return False
        return False

    def reconnect(self):
        """重新连接数据库"""
        self.disconnect()
        return self.connect()

    @contextmanager
    def get_cursor(self):
        """
        获取游标的上下文管理器，自动处理连接和事务

        Yields:
            cursor: 数据库游标
        """
        cursor = None
        try:
            # 确保连接有效
            if not self.is_connected():
                self.connect()

            cursor = self.connection.cursor()
            yield cursor
            self.connection.commit()
        except Exception as e:
            if self.connection:
                self.connection.rollback()
            logger.error(f"数据库操作失败: {e}")
            raise e
        finally:
            if cursor:
                cursor.close()

    def execute_query(self, sql: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        执行查询语句

        Args:
            sql: SQL查询语句
            params: 参数元组

        Returns:
            List[Dict]: 查询结果列表
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute(sql, params)
                result = cursor.fetchall()
                logger.info(f"查询执行成功，返回 {len(result)} 条记录")
                return result
        except Exception as e:
            logger.error(f"查询执行失败: {e}")
            return []

    def execute_update(self, sql: str, params: Optional[tuple] = None) -> int:
        """
        执行更新操作（INSERT, UPDATE, DELETE）

        Args:
            sql: SQL语句
            params: 参数元组

        Returns:
            int: 受影响的行数
        """
        try:
            with self.get_cursor() as cursor:
                affected_rows = cursor.execute(sql, params)
                logger.info(f"更新操作执行成功，影响 {affected_rows} 行")
                return affected_rows
        except Exception as e:
            logger.error(f"更新操作执行失败: {e}")
            return 0

    def execute_many(self, sql: str, params_list: List[tuple]) -> int:
        """
        批量执行操作

        Args:
            sql: SQL语句
            params_list: 参数列表

        Returns:
            int: 受影响的总行数
        """
        try:
            with self.get_cursor() as cursor:
                affected_rows = cursor.executemany(sql, params_list)
                logger.info(f"批量操作执行成功，影响 {affected_rows} 行")
                return affected_rows
        except Exception as e:
            logger.error(f"批量操作执行失败: {e}")
            return 0

    def insert_emotion_type(self, emotion_code: str, emotion_name: str,
                            description: str, color_code: str) -> bool:
        """
        插入情绪类型数据（示例方法）

        Args:
            emotion_code: 情绪编码
            emotion_name: 情绪名称
            description: 描述
            color_code: 颜色代码

        Returns:
            bool: 是否成功
        """
        sql = """
              INSERT INTO emotion_types (emotion_code, emotion_name, description, color_code)
              VALUES (%s, %s, %s, %s) \
              """
        params = (emotion_code, emotion_name, description, color_code)

        try:
            affected_rows = self.execute_update(sql, params)
            return affected_rows > 0
        except Exception as e:
            logger.error(f"插入情绪类型失败: {e}")
            return False

    def get_all_emotion_types(self) -> List[Dict[str, Any]]:
        """
        获取所有情绪类型（示例方法）

        Returns:
            List[Dict]: 情绪类型列表
        """
        sql = "SELECT * FROM emotion_types ORDER BY emotion_code"
        return self.execute_query(sql)

    def insert_audio_record(self, speaker_id: int, speech_time: str,
                            speech_content: str, emotion: str,
                            emotion_confidence: float, recognition_confidence: float,
                            audio_file_path: str = None, audio_duration: float = None) -> bool:
        """
        插入音频识别记录（示例方法）

        Args:
            speaker_id: 说话人ID
            speech_time: 说话时间
            speech_content: 说话内容
            emotion: 情绪类型
            emotion_confidence: 情绪识别置信度
            recognition_confidence: 语音识别置信度
            audio_file_path: 音频文件路径
            audio_duration: 音频时长

        Returns:
            bool: 是否成功
        """
        sql = """
              INSERT INTO audio_recognition_records
              (speaker_id, speaker_code, speaker_name, speech_time, speech_content,
               emotion, emotion_confidence, recognition_confidence, audio_file_path, audio_duration)
              VALUES (%s, (SELECT user_code FROM user_voiceprints WHERE id = %s),
                      (SELECT user_name FROM user_voiceprints WHERE id = %s),
                      %s, %s, %s, %s, %s, %s, %s) \
              """
        params = (speaker_id, speaker_id, speaker_id, speech_time, speech_content,
                  emotion, emotion_confidence, recognition_confidence,
                  audio_file_path, audio_duration)

        try:
            affected_rows = self.execute_update(sql, params)
            return affected_rows > 0
        except Exception as e:
            logger.error(f"插入音频记录失败: {e}")
            return False



