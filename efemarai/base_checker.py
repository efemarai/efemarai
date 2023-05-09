import traceback

from efemarai.console import console


class BaseChecker:
    def __init__(self, print_warnings=True):
        self._print_warnings = print_warnings

    def print_warnings(self, print_warnings):
        self._print_warnings = print_warnings

    @staticmethod
    def _error(message, print_exception=False):
        console.print(f":poop: {message}", style="red")

        if print_exception:
            traceback.print_exc()

        raise AssertionError()

    def _warning(self, message):
        if self._print_warnings:
            console.print(f":face_with_monocle: {message}", style="orange1")

    def _get_required_item(self, definition, key, parent=None):
        item = definition.get(key)

        if item is None:
            message = f"Missing field '{key}'"
            if parent is not None:
                message += f" (in '{parent}')"
            self._error(message)

        return item
