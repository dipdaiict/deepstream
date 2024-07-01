import os
import sys
import traceback

def error_message_detail(error: Exception) -> str:
    exc_type, exc_obj, exc_tb = sys.exc_info()
    file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    error_message = f"Error occurred in file '{file_name}', line {exc_tb.tb_lineno}: {str(error)}"
    return error_message

class DataException(Exception):
    def __init__(self, error_message):
        """
        :param error_message: error message in string format
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message)

    def __str__(self):
        return self.error_message