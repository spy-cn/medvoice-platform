
CREATE DATABASE IF NOT EXISTS `medvoice_identity` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE `medvoice_identity`;

-- 用户声纹表
CREATE TABLE `user_voiceprints` (
    `id` INT PRIMARY KEY AUTO_INCREMENT COMMENT '用户ID',
    `user_code` VARCHAR(50) UNIQUE NOT NULL COMMENT '用户编码',
    `user_name` VARCHAR(100) NOT NULL COMMENT '用户名称',
    `voiceprint_data` LONGTEXT NOT NULL COMMENT '声纹特征数据',
    `voice_sample_count` INT DEFAULT 0 COMMENT '声纹样本数量',
    `registration_time` DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '注册时间',
    `last_updated` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '最后更新时间',
    `status` TINYINT DEFAULT 1 COMMENT '状态：0-禁用，1-启用',
    `remarks` TEXT COMMENT '备注信息',
    `crt` DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `upt` DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX `idx_user_code` (`user_code`),
    INDEX `idx_user_name` (`user_name`),
    INDEX `idx_status` (`status`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户声纹库';

-- 音频识别记录表
CREATE TABLE `audio_recognition_records` (
    `id` BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '记录ID',
    `speaker_id` INT NOT NULL COMMENT '说话人ID',
    `speaker_code` VARCHAR(50) NOT NULL COMMENT '说话人编码',
    `speaker_name` VARCHAR(100) NOT NULL COMMENT '说话人名称',
    `speech_time` DATETIME NOT NULL COMMENT '说话时间',
    `speech_content` TEXT NOT NULL COMMENT '说话内容',
    `emotion` VARCHAR(20) COMMENT '情绪类型：happy, sad, angry, neutral, excited, calm等',
    `emotion_confidence` DECIMAL(5,4) COMMENT '情绪识别置信度',
    `audio_file_path` VARCHAR(500) COMMENT '音频文件路径',
    `audio_duration` DECIMAL(8,3) COMMENT '音频时长(秒)',
    `recognition_confidence` DECIMAL(5,4) COMMENT '语音识别置信度',
    `recognition_time` DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '识别时间',
    `crt` DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `upt` DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX `idx_speaker_id` (`speaker_id`),
    INDEX `idx_speaker_code` (`speaker_code`),
    INDEX `idx_speech_time` (`speech_time`),
    INDEX `idx_emotion` (`emotion`),
    INDEX `idx_recognition_time` (`recognition_time`),
    FOREIGN KEY (`speaker_id`) REFERENCES `user_voiceprints`(`id`) ON DELETE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='音频识别记录';

-- 情绪类型字典表
CREATE TABLE `emotion_types` (
    `emotion_code` VARCHAR(20) PRIMARY KEY COMMENT '情绪编码',
    `emotion_name` VARCHAR(50) NOT NULL COMMENT '情绪名称',
    `description` VARCHAR(200) COMMENT '情绪描述',
    `color_code` VARCHAR(7) COMMENT '颜色代码',
    `is_active` TINYINT DEFAULT 1 COMMENT '是否启用'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='情绪类型字典';

-- 插入默认的情绪类型数据
INSERT INTO `emotion_types` (`emotion_code`, `emotion_name`, `description`, `color_code`) VALUES
('happy', '开心', '高兴、愉快的情绪', '#FFD700'),
('sad', '难过', '伤心、难过的情绪', '#4169E1'),
('angry', '生气', '生气、愤怒的情绪', '#FF4500'),
('neutral', '中性', '平静、无显著情绪', '#808080'),
('fearful', '恐惧', '害怕、恐惧的情绪', '#8B4513'),
('surprised', '惊讶', '惊奇、意外的情绪', '#FFA500'),
('disgusted', '厌恶', '讨厌、反感的情绪', '#9ACD32'),
('other', '其他', '其他未分类的情绪', '#A9A9A9');