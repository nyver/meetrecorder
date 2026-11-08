"""Meeting Recorder — запись экрана и звука, транскрипция, протокол и summary."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("meeting-recorder")
except PackageNotFoundError:  # запуск из исходников без установки
    __version__ = "0.0.0+dev"
