import pytest
from unittest.mock import Mock, patch, MagicMock
from io import BytesIO
import os
from main import extract_audio, split_audio, speech_to_text, summarize


class TestExtractAudio:
    """Тесты для функции извлечения аудио"""
    
    @patch('main.AudioSegment')
    def test_extract_audio_success(self, mock_audio_segment):
        """Тест успешного извлечения аудио"""
        # Arrange
        mock_file = Mock()
        mock_file.name = "test.mp4"
        mock_audio = Mock()
        mock_audio_segment.from_file.return_value = mock_audio
        
        # Act
        result = extract_audio(mock_file)
        
        # Assert
        assert isinstance(result, BytesIO)
        mock_audio_segment.from_file.assert_called_once_with(mock_file)
        mock_audio.export.assert_called_once_with(result, format="mp3")


class TestSplitAudio:
    """Тесты для функции разделения аудио на чанки"""
    
    @patch('main.AudioSegment')
    @patch('main.silence')
    @patch('main.utils')
    def test_split_audio_by_silence(self, mock_utils, mock_silence, mock_audio_segment):
        """Тест разделения аудио по тишине"""
        # Arrange
        audio_buffer = b"fake_audio_data"
        number_of_chunks = 3
        
        mock_segment = Mock()
        mock_audio_segment.from_mp3.return_value = mock_segment
        mock_segment.__len__ = Mock(return_value=30000)  # 30 seconds
        
        mock_chunks = [Mock(), Mock(), Mock()]
        mock_silence.split_on_silence.return_value = mock_chunks
        
        # Act
        result = split_audio(audio_buffer, number_of_chunks)
        
        # Assert
        assert len(result) == 3
        mock_audio_segment.from_mp3.assert_called_once()
        mock_silence.split_on_silence.assert_called_once_with(
            mock_segment, min_silence_len=1000, silence_thresh=-16, keep_silence=200
        )
    
    @patch('main.AudioSegment')
    @patch('main.silence')
    @patch('main.utils')
    def test_split_audio_too_few_chunks(self, mock_utils, mock_silence, mock_audio_segment):
        """Тест разделения аудио при недостатке чанков"""
        # Arrange
        audio_buffer = b"fake_audio_data"
        number_of_chunks = 5
        
        mock_segment = Mock()
        mock_audio_segment.from_mp3.return_value = mock_segment
        mock_segment.__len__ = Mock(return_value=50000)  # 50 seconds
        
        # Слишком мало чанков по тишине
        mock_chunks = [Mock()]
        mock_silence.split_on_silence.return_value = mock_chunks
        
        # Чанки по времени
        mock_time_chunks = [Mock(), Mock(), Mock(), Mock(), Mock()]
        mock_utils.make_chunks.return_value = mock_time_chunks
        
        # Act
        result = split_audio(audio_buffer, number_of_chunks)
        
        # Assert
        assert len(result) == 5
        mock_utils.make_chunks.assert_called_once()


class TestSpeechToText:
    """Тесты для функции распознавания речи"""
    
    @patch('main.OpenAI')
    def test_speech_to_text_small_file(self, mock_openai):
        """Тест распознавания речи для маленького файла"""
        # Arrange
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        mock_transcript = Mock()
        mock_transcript.text = "Тестовый текст"
        mock_client.audio.transcriptions.create.return_value = mock_transcript
        
        audio_file = BytesIO(b"fake_audio_data")
        
        # Act
        result = speech_to_text(audio_file)
        
        # Assert
        assert result == "Тестовый текст"
        mock_client.audio.transcriptions.create.assert_called_once()
    
    @patch('main.split_audio')
    @patch('main.OpenAI')
    def test_speech_to_text_large_file(self, mock_openai, mock_split_audio):
        """Тест распознавания речи для большого файла"""
        # Arrange
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        mock_transcript1 = Mock()
        mock_transcript1.text = "Первая часть"
        mock_transcript2 = Mock()
        mock_transcript2.text = "Вторая часть"
        mock_client.audio.transcriptions.create.side_effect = [mock_transcript1, mock_transcript2]
        
        # Создаем большой файл
        large_audio_data = b"x" * (26 * 1024 * 1024)  # Больше лимита
        audio_file = BytesIO(large_audio_data)
        
        mock_split_audio.return_value = [b"chunk1", b"chunk2"]
        
        # Act
        result = speech_to_text(audio_file)
        
        # Assert
        assert result == "Первая часть\nВторая часть"
        assert mock_client.audio.transcriptions.create.call_count == 2


class TestSummarize:
    """Тесты для функции создания саммари"""
    
    @patch('main.OpenAI')
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    def test_summarize_success(self, mock_openai):
        """Тест успешного создания саммари"""
        # Arrange
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Тестовое саммари"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        
        conversation = "Тестовый разговор"
        system_prompt = "Системный промпт"
        prompt = "Промпт {conversation}"
        
        # Act
        result = summarize(conversation, system_prompt, prompt)
        
        # Assert
        assert result == "Тестовое саммари"
        mock_client.chat.completions.create.assert_called_once()


class TestIntegration:
    """Интеграционные тесты"""
    
    def test_environment_variables(self):
        """Тест наличия необходимых переменных окружения"""
        # Проверяем, что API_KEY установлен (даже если это заглушка)
        from main import API_KEY
        assert API_KEY is not None
    
    def test_constants_defined(self):
        """Тест определения констант"""
        from main import (
            SPEECH_TO_TEXT_MODEL,
            AUDIO_FILE_SIZE_LIMIT,
            SUMMARIZE_MODEL,
            DEFAULT_SYSTEM_PROMPT,
            DEFAULT_PROMPT
        )
        
        assert SPEECH_TO_TEXT_MODEL == "whisper-1"
        assert AUDIO_FILE_SIZE_LIMIT == 25 * 1024 * 1024
        assert SUMMARIZE_MODEL == "gpt-4-1106-preview"
        assert isinstance(DEFAULT_SYSTEM_PROMPT, str)
        assert isinstance(DEFAULT_PROMPT, str)
        assert "{conversation}" in DEFAULT_PROMPT


if __name__ == "__main__":
    pytest.main([__file__])