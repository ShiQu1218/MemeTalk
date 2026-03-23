from memetalk.cli.main import build_parser


def test_cli_parser_supports_telegram_run_command() -> None:
    args = build_parser().parse_args(["telegram", "run"])

    assert args.command == "telegram"
    assert args.telegram_command == "run"


def test_cli_parser_supports_telegram_autostart_command() -> None:
    args = build_parser().parse_args(["telegram", "should-autostart"])

    assert args.command == "telegram"
    assert args.telegram_command == "should-autostart"
