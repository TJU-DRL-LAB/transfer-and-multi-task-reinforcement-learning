import logging
import sys
import os
from logging.handlers import SocketHandler


class LabelFilter(logging.Filter):

    def filter(self, record):
        if not hasattr(record, 'label'):
            record.label = 'None'
        return True


class EnvLogger:

    def __init__(self, name='env', log_level=logging.INFO):
        self._logger = logging.getLogger(name)
        self._logger.setLevel(log_level)
        self._formatter = logging.Formatter(
            '%(levelname)s:%(name)s:%(asctime)s:%(label)s:%(message)s')
        self._logger.addFilter(LabelFilter())

    def add_socket_handler(self, ip, port):
        hd = SocketHandler(ip, port)
        self._add_handler(hd)

    def add_file_handler(self, dir_log, f_name):
        hd = logging.FileHandler(os.path.join(dir_log, f_name))
        self._add_handler(hd)

    def add_stream_handler(self):
        hd = logging.StreamHandler(sys.stdout)
        self._add_handler(hd)

    def _add_handler(self, handler):
        handler.setFormatter(self._formatter)
        self._logger.addHandler(handler)

    @property
    def logger(self):
        return self._logger


if __name__ == '__main__':
    envLogger = EnvLogger()
    envLogger.add_stream_handler()
    envLogger.add_file_handler('.', 'log.txt')
    # To use the socket handler, the log-server.py needs to be
    # executed first.
    # envLogger.add_socket_handler('localhost',
    #                              logging.handlers.DEFAULT_TCP_LOGGING_PORT)
    logger = envLogger.logger
    import coloredlogs
    coloredlogs.install(level='DEBUG', logger=logger)

    reward = 10
    logger.warning(reward, extra={'label': 'reward'})
    logger.info(reward, extra={'label': 'reward'})
    logger.error(reward, extra={'label': 'reward'})
