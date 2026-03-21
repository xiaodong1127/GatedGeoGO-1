import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler


class Logger:
    """
    Logger class
    """

    def __init__(self):
        """
        Init function
        """
        self.pre()
        self.name = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.__all_log_path = os.path.join(self.log_path, f"{self.name}.log")
        self.__logger = logging.getLogger()
        self.__logger.setLevel(logging.DEBUG)

    def pre(self):
        exec_path = os.getcwd()
        self.log_path = os.path.join(exec_path, 'logs')
        if not os.path.exists(self.log_path): os.mkdir(self.log_path)

    @staticmethod
    def __init_logger_handler(log_path):
        logger_handler = RotatingFileHandler(filename=log_path, maxBytes=5 * 1024 * 1024, backupCount=3,
                                             encoding='utf-8')
        return logger_handler

    def __set_log_handler(self, logger_handler, level=logging.DEBUG):
        logger_handler.setLevel(level=level)
        self.__logger.addHandler(logger_handler)

    @staticmethod
    def __set_log_formatter(file_handler):
        formatter = logging.Formatter('%(asctime)s-[Log information]: %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')
        file_handler.setFormatter(formatter)

    @staticmethod
    def __close_handler(file_handler):
        file_handler.close()

    def __console(self, level, message):
        all_logger_handler = self.__init_logger_handler(self.__all_log_path)
        self.__set_log_formatter(all_logger_handler)
        self.__set_log_handler(all_logger_handler)

        if level == 'info':
            self.__logger.info(message)
        elif level == 'debug':
            self.__logger.debug(message)
        elif level == 'warning':
            self.__logger.warning(message)
        elif level == 'error':
            self.__logger.error(message)
        elif level == 'critical':
            self.__logger.critical(message)

        self.__logger.removeHandler(all_logger_handler)

        self.__close_handler(all_logger_handler)

    def do_debug(self, message=''):
        self.__console('debug', message)

    def do_info(self, message=''):
        self.__console('info', message)

    def do_warning(self, message=''):
        self.__console('warning', message)

    def do_error(self, message=''):
        self.__console('error', message)

    def do_critical(self, message=''):
        self.__console('critical', message)

    def do_print(self, message=''):
        print(message)
        self.__console('info', message)


log = Logger()


def logging_params(func):
    """
    Logging function params.
    :return:
    """

    def wrapper(self, *args, **kwargs):
        for k, v in kwargs.items():
            log.do_print(f'func: {func.__name__} {k}: {v}')
        return func(self, *args, **kwargs)

    return wrapper
