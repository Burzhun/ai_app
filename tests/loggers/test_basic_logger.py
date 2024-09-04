import logging
import pytest
from src.loggers.basic_logger import Logger


@pytest.fixture
def log_file(tmp_path):
    return tmp_path / "test.log"


@pytest.fixture
def basic_logger(log_file):
    return Logger("test_logger", log_file).get_logger()


def test_log_messages_to_file_and_stdout(basic_logger, caplog, log_file):
    test_message = "This is a test message"

    with caplog.at_level(logging.DEBUG):
        basic_logger.debug(test_message)

    # Проверяем, что сообщение записано в caplog
    assert test_message in caplog.text
    assert "DEBUG" in caplog.text
    assert "test_logger" in caplog.text

    # Проверяем, что сообщение записано в файл
    with open(log_file, "r") as f:
        log_contents = f.read()
        assert test_message in log_contents
        assert "DEBUG" in log_contents
        assert "test_logger" in log_contents
